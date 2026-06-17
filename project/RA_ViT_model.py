import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from data_splitting import create_vit_b16_dataloaders


NUM_BIRD_CLASSES = 200
VIT_B16_FEATURE_DIM = 768
VIT_B16_IMAGE_SIZE = 224
VIT_B16_PATCH_SIZE = 16
VIT_B16_PATCH_GRID = VIT_B16_IMAGE_SIZE // VIT_B16_PATCH_SIZE
LOCAL_CROP_SIZE = 7 * VIT_B16_PATCH_SIZE
DEFAULT_RA_VIT_CHECKPOINT_PATH = "ra_vit_last_epoch.pt"


def build_fc_classifier(num_classes=NUM_BIRD_CLASSES, fc1_dim=512, dropout=0.3):
    return nn.Sequential(
        nn.Linear(VIT_B16_FEATURE_DIM, fc1_dim),
        nn.BatchNorm1d(fc1_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc1_dim, NUM_BIRD_CLASSES),
        )


def build_pretrained_vit_b16_backbone(freeze_backbone=True):
    backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    backbone.heads = nn.Identity()

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    return backbone


def extract_attention_map(vit_backbone, images, layer_index=-1, average_heads=True):
    """
    Extract CLS-token attention to image patches from a torchvision ViT backbone.

    Returns a tensor with shape [batch_size, 14, 14] for ViT-B/16 on 224x224 images.
    """
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

    if average_heads:
        return class_to_patch_attention.reshape(batch_size, VIT_B16_PATCH_GRID, VIT_B16_PATCH_GRID)

    num_heads = class_to_patch_attention.shape[1]
    return class_to_patch_attention.reshape(
        batch_size,
        num_heads,
        VIT_B16_PATCH_GRID,
        VIT_B16_PATCH_GRID,
    )


def crop_attention_zoom(images, attention_maps, crop_size=LOCAL_CROP_SIZE, output_size=VIT_B16_IMAGE_SIZE):
    """
    Crop a 7-patch by 7-patch square around the highest-attention 3x3 patch area.

    images should have shape [batch_size, 3, 224, 224].
    attention_maps should have shape [batch_size, 14, 14].
    """
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
            torch.ones(1, 1, 3, 3, device=attention_map.device, dtype=attention_map.dtype),
        ).squeeze(0).squeeze(0)
        flat_index = window_scores.argmax().item()
        window_top = flat_index // window_scores.shape[-1]
        window_left = flat_index % window_scores.shape[-1]
        patch_row = window_top + 1
        patch_col = window_left + 1

        center_y = math.floor((patch_row + 0.5) * patch_size)
        center_x = math.floor((patch_col + 0.5) * patch_size)
        padded_top = center_y
        padded_left = center_x

        crop = padded_images[
            image_index:image_index + 1,
            :,
            padded_top:padded_top + crop_size,
            padded_left:padded_left + crop_size,
        ]
        local_crops.append(crop)

    local_crops = torch.cat(local_crops, dim=0)
    local_crops = F.interpolate(
        local_crops,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    return local_crops


class RA_ViT(nn.Module):
    """
    Recurrent-attention style ViT with global and local classification streams.

    The global stream classifies the full image. The local stream uses the global
    stream attention map to crop a 7x7 patch region, resizes it to 224x224, and
    classifies that zoomed crop with a second ViT-B/16 backbone.
    """

    def __init__(
        self,
        num_classes=NUM_BIRD_CLASSES,
        fc1_dim=512,
        dropout=0.3,
        freeze_backbones=True,
        attention_layer_index=-1,
        local_crop_size=LOCAL_CROP_SIZE,
    ):
        super(RA_ViT, self).__init__()
        self.name = "RA_ViT"
        self.attention_layer_index = attention_layer_index
        self.local_crop_size = local_crop_size

        self.global_vit = build_pretrained_vit_b16_backbone(freeze_backbone=freeze_backbones)
        self.local_vit = build_pretrained_vit_b16_backbone(freeze_backbone=freeze_backbones)
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
            local_images = crop_attention_zoom(
                images,
                attention_maps,
                crop_size=self.local_crop_size,
                output_size=VIT_B16_IMAGE_SIZE,
            )

        local_features = self.local_vit(local_images)
        local_logits = self.local_classifier(local_features)

        if return_attention:
            return global_logits, local_logits, attention_maps, local_images

        total_logit = global_logits + local_logits
        return global_logits, local_logits, total_logit


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_ra_vit_epoch_metrics(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_global_correct = 0
    total_local_correct = 0
    #here!
    total_summed_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            #here!
            global_logits, local_logits, total_logits = model(images)
            global_loss = criterion(global_logits, labels)
            local_loss = criterion(local_logits, labels)
            loss = global_loss + local_loss

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_global_correct += (global_logits.argmax(dim=1) == labels).sum().item()
            total_local_correct += (local_logits.argmax(dim=1) == labels).sum().item()
            #here!
            total_summed_correct += (total_logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    global_accuracy = total_global_correct / total_examples
    local_accuracy = total_local_correct / total_examples
    #here!
    total_accuracy = total_summed_correct / total_examples
    #here!
    return average_loss, global_accuracy, local_accuracy, total_accuracy


def train_ra_vit(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_path=DEFAULT_RA_VIT_CHECKPOINT_PATH,
    device=None,
):
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.0001,
    )

    history = {
        "train_loss": [],
        "train_global_accuracy": [],
        "train_local_accuracy": [],
        "val_loss": [],
        "val_global_accuracy": [],
        "val_local_accuracy": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        #here!
        train_loss, train_global_accuracy, train_local_accuracy, train_total_acc = calculate_ra_vit_epoch_metrics(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        #here!
        val_loss, val_global_accuracy, val_local_accuracy, val_total_acc = calculate_ra_vit_epoch_metrics(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["train_global_accuracy"].append(train_global_accuracy)
        history["train_local_accuracy"].append(train_local_accuracy)
        history["val_loss"].append(val_loss)
        history["val_global_accuracy"].append(val_global_accuracy)
        history["val_local_accuracy"].append(val_local_accuracy)

#here!
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, "
            f"train global acc: {train_global_accuracy:.4f}, "
            f"train local acc: {train_local_accuracy:.4f}, "
            f"train total acca: {train_total_acc:.4f}, "
            f"val loss: {val_loss:.4f}, "
            f"val global acc: {val_global_accuracy:.4f}, "
            f"val local acc: {val_local_accuracy:.4f}, "
            f"val total acc: {val_total_acc:.4f}"
        )

    elapsed_time = time.perf_counter() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved RA_ViT parameters from last epoch to {checkpoint_path}")

    return model, history


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    fc1_dim = 512
    dropout = 0.3
    seed = 1
    device = get_device()

    train_loader, val_loader, test_loader, class_names = create_vit_b16_dataloaders(
        batch_size=batch_size,
        seed=seed,
    )
    model = RA_ViT(
        num_classes=len(class_names),
        fc1_dim=fc1_dim,
        dropout=dropout,
        freeze_backbones=True,
    )
    trained_model, history = train_ra_vit(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        checkpoint_path=DEFAULT_RA_VIT_CHECKPOINT_PATH,
        device=device,
    )
