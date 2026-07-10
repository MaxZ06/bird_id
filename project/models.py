import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


NUM_BIRD_CLASSES = 200
VIT_B16_FEATURE_DIM = 768
VIT_B16_IMAGE_SIZE = 224
VIT_B16_PATCH_SIZE = 16
VIT_B16_PATCH_GRID = VIT_B16_IMAGE_SIZE // VIT_B16_PATCH_SIZE
LOCAL_CROP_SIZE = 7 * VIT_B16_PATCH_SIZE


def build_pretrained_vit_b16_backbone(freeze_backbone=True):
    backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    backbone.heads = nn.Identity()

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    return backbone


def build_fc_classifier(num_classes=NUM_BIRD_CLASSES, fc1_dim=512, dropout=0.3):
    return nn.Sequential(
        nn.Linear(VIT_B16_FEATURE_DIM, fc1_dim),
        nn.BatchNorm1d(fc1_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc1_dim, num_classes),
    )


def extract_attention_map(vit_backbone, images, layer_index=-1, average_heads=True):
    tokens = vit_backbone._process_input(images)
    batch_size = tokens.shape[0]
    class_token = vit_backbone.class_token.expand(batch_size, -1, -1)
    tokens = torch.cat((class_token, tokens), dim=1)

    encoder = vit_backbone.encoder
    tokens = encoder.dropout(tokens)
    layers = list(encoder.layers.children())
    selected_layer_index = layer_index % len(layers)
    selected_attention = None

    for index, layer in enumerate(layers):
        attention_input = layer.ln_1(tokens)
        attention_output, attention_weights = layer.self_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + layer.dropout(attention_output)
        tokens = tokens + layer.mlp(layer.ln_2(tokens))

        if index == selected_layer_index:
            selected_attention = attention_weights

    if selected_attention is None:
        raise RuntimeError("Could not extract ViT attention weights.")

    class_to_patch_attention = selected_attention[:, :, 0, 1:]
    if average_heads:
        class_to_patch_attention = class_to_patch_attention.mean(dim=1)
        return class_to_patch_attention.reshape(
            batch_size,
            VIT_B16_PATCH_GRID,
            VIT_B16_PATCH_GRID,
        )

    num_heads = class_to_patch_attention.shape[1]
    return class_to_patch_attention.reshape(
        batch_size,
        num_heads,
        VIT_B16_PATCH_GRID,
        VIT_B16_PATCH_GRID,
    )


def attention_crop(
    images,
    attention_maps,
    crop_size=LOCAL_CROP_SIZE,
    output_size=VIT_B16_IMAGE_SIZE,
):
    batch_size, channels, height, width = images.shape
    if height != width:
        raise ValueError("RA_ViT expects square image tensors.")

    patch_size = height // attention_maps.shape[-1]
    half_crop = crop_size // 2
    padded_images = F.pad(images, (half_crop, half_crop, half_crop, half_crop))
    local_crops = []

    for image_index in range(batch_size):
        attention_map = attention_maps[image_index]
        window_scores = F.conv2d(
            attention_map.unsqueeze(0).unsqueeze(0),
            torch.ones(
                1,
                1,
                3,
                3,
                device=attention_map.device,
                dtype=attention_map.dtype,
            ),
        ).squeeze(0).squeeze(0)
        flat_index = window_scores.argmax().item()
        window_top = flat_index // window_scores.shape[-1]
        window_left = flat_index % window_scores.shape[-1]
        patch_row = window_top + 1
        patch_col = window_left + 1

        center_y = math.floor((patch_row + 0.5) * patch_size)
        center_x = math.floor((patch_col + 0.5) * patch_size)

        crop = padded_images[
            image_index:image_index + 1,
            :,
            center_y:center_y + crop_size,
            center_x:center_x + crop_size,
        ]
        local_crops.append(crop)

    local_crops = torch.cat(local_crops, dim=0)
    return F.interpolate(
        local_crops,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )

class weighted_logit_combiner(nn.Module):
    def __init__(self, init_w1=0.3):
        super().__init__()
        self.name = "weightedLogitCombiner"

        init_w1 = torch.tensor(init_w1)
        self.raw_w1 = nn.Parameter(torch.logit(init_w1))

    def forward(self, global_logits, local_logits):
        w1 = torch.sigmoid(self.raw_w1)
        total_logits = (1 - w1) * global_logits + w1 * local_logits
        return total_logits


class linear_combiner(nn.Module):
    def __init__(self, summed_logits=400, out_logits=200):
        super().__init__()
        self.name = "linearCombiner"
        self.fcs = nn.Linear(summed_logits, out_logits)

    def forward(self, logits):
        out = self.fcs(logits)
        return out

class RA_ViT(nn.Module):
    def __init__(
        self,
        num_classes=NUM_BIRD_CLASSES,
        fc1_dim=512,
        dropout=0.3,
        freeze_backbones=True,
        attention_layer_index=-1,
        local_crop_size=LOCAL_CROP_SIZE,
    ):
        super().__init__()
        self.name = "RA_ViT"
        self.attention_layer_index = attention_layer_index
        self.local_crop_size = local_crop_size

        self.global_vit = build_pretrained_vit_b16_backbone(
            freeze_backbone=freeze_backbones,
        )
        self.local_vit = build_pretrained_vit_b16_backbone(
            freeze_backbone=freeze_backbones,
        )
        self.global_classifier = build_fc_classifier(
            num_classes=num_classes,
            fc1_dim=fc1_dim,
            dropout=dropout,
        )
        self.local_classifier = build_fc_classifier(
            num_classes=num_classes,
            fc1_dim=fc1_dim,
            dropout=dropout,
        )

    def forward(self, images, return_attention=False):
        global_features = self.global_vit(images)
        global_logits = self.global_classifier(global_features)

        with torch.no_grad():
            attention_maps = extract_attention_map(
                self.global_vit,
                images,
                layer_index=self.attention_layer_index,
                average_heads=True,
            )
            local_images = attention_crop(
                images,
                attention_maps,
                crop_size=self.local_crop_size,
                output_size=VIT_B16_IMAGE_SIZE,
            )

        local_features = self.local_vit(local_images)
        local_logits = self.local_classifier(local_features)

        
#        total_logits = global_logits + local_logits

        if return_attention:
            return global_logits, local_logits, local_images

        return global_logits, local_logits
