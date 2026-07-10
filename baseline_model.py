import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from project.data_splitting import create_vit_b16_dataloaders


NUM_BIRD_CLASSES = 200
VIT_B16_FEATURE_DIM = 768


class SimpleViTB16(nn.Module):
    def __init__(
        self,
        num_classes=NUM_BIRD_CLASSES,
        hidden_dim=512,
        dropout=0.3,
        weights=ViT_B_16_Weights.DEFAULT,
    ):
        super().__init__()
        self.name = "SimpleViTB16"

        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(VIT_B16_FEATURE_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images):
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(optimizer, parameters, learning_rate):
    if isinstance(optimizer, str):
        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate)
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer(parameters, lr=learning_rate)


def get_loss(loss):
    if isinstance(loss, str):
        loss_name = loss.lower()
        if loss_name in ("ce", "cross_entropy", "crossentropy"):
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported loss: {loss}")

    return loss


def save_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def calculate_epoch_metrics(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top_3_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            top_3_predictions = logits.topk(3, dim=1).indices
            total_top_3_correct += (
                top_3_predictions == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "top_3_acc": total_top_3_correct / total_examples,
    }


def train_simple_vit_b16(
    batch_size=32,
    learning_rate=0.001,
    epochs=5,
    optimizer="adam",
    loss="ce",
    model=None,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path="checkpoints/simple_vit_b16.pt",
):
    device = device or get_device()
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }
    if data_root is not None:
        dataloader_kwargs["data_root"] = data_root

    train_loader, val_loader, test_loader, class_names = create_vit_b16_dataloaders(
        **dataloader_kwargs,
    )

    model = model or SimpleViTB16(num_classes=len(class_names))
    model.to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = get_optimizer(optimizer, trainable_parameters, learning_rate)
    criterion = get_loss(loss)

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = calculate_epoch_metrics(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = calculate_epoch_metrics(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train acc: {train_metrics['accuracy']:.4f}, "
            f"train top 3 acc: {train_metrics['top_3_acc']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val acc: {val_metrics['accuracy']:.4f}, "
            f"val top 3 acc: {val_metrics['top_3_acc']:.4f}"
        )

    elapsed_seconds = time.time() - start_time
    checkpoint_path = save_checkpoint(model, checkpoint_path)

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    train_simple_vit_b16()
