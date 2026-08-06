import sys
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import create_vit_b16_dataloaders


NUM_BIRD_CLASSES = 200
RESNET50_FEATURE_DIM = 2048


class SimpleResNet50(nn.Module):
    def __init__(
        self,
        num_classes=NUM_BIRD_CLASSES,
        hidden_dim=512,
        dropout=0.3,
        weights=ResNet50_Weights.DEFAULT,
    ):
        super().__init__()
        self.name = "SimpleResNet50"

        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(RESNET50_FEATURE_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images):
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits


def get_trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_resnet50_fine_tuning_trainable(model, num_unfrozen_layers=1):
    """Train the classifier and the final N ResNet50 residual blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    residual_blocks = [
        block
        for stage in (
            model.backbone.layer1,
            model.backbone.layer2,
            model.backbone.layer3,
            model.backbone.layer4,
        )
        for block in stage.children()
    ]
    if not 1 <= num_unfrozen_layers <= len(residual_blocks):
        raise ValueError(
            "num_unfrozen_layers must be between 1 and "
            f"{len(residual_blocks)}."
        )

    for block in residual_blocks[-num_unfrozen_layers:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    return len(residual_blocks)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(optimizer, parameters, learning_rate, weight_decay=1e-4):
    if isinstance(optimizer, str):
        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


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


def get_epoch_checkpoint_path(checkpoint_path, epoch):
    """Add an epoch suffix while preserving the checkpoint file extension."""
    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix or ".pt"
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_e{epoch}{suffix}"
    )


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

def train_simple_resnet50(
    batch_size=32,
    learning_rate=0.001,
    fine_tune=False,
    backbone_learning_rate=1e-5,
    classifier_learning_rate=None,
    num_unfrozen_layers=1,
    weight_decay=1e-4,
    epochs=5,
    optimizer="adam",
    loss="ce",
    model=None,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path=Path(__file__).resolve().parents[2] / "checkpoints" / "simple_resnet50.pt",
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

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

    model = model or SimpleResNet50(num_classes=len(class_names))
    model.to(device)

    if classifier_learning_rate is None:
        classifier_learning_rate = learning_rate
    unfrozen_backbone_layers = 0
    if fine_tune:
        set_resnet50_fine_tuning_trainable(
            model,
            num_unfrozen_layers=num_unfrozen_layers,
        )
        unfrozen_backbone_layers = num_unfrozen_layers
        classifier_parameters = [
            parameter
            for parameter in model.classifier.parameters()
            if parameter.requires_grad
        ]
        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        trainable_parameters = [
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
            },
            {
                "params": classifier_parameters,
                "lr": classifier_learning_rate,
            },
        ]
        optimizer_learning_rate = classifier_learning_rate
        print(
            f"Fine-tuning the last {num_unfrozen_layers} ResNet50 residual "
            f"block(s) at lr={backbone_learning_rate:g}; classifier "
            f"lr={classifier_learning_rate:g}."
        )
    else:
        trainable_parameters = get_trainable_parameters(model)
        optimizer_learning_rate = learning_rate

    optimizer = get_optimizer(
        optimizer,
        trainable_parameters,
        optimizer_learning_rate,
        weight_decay=weight_decay,
    )
    criterion = get_loss(loss)

    history = []
    best_val_accuracy = float("-inf")
    best_checkpoint_path = None
    start_time = time.time()

    for epoch in range(epochs):
        current_epoch = epoch + 1
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
            "epoch": current_epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_checkpoint_path = save_checkpoint(
                model,
                get_epoch_checkpoint_path(checkpoint_path, current_epoch),
            )
            print(
                f"Saved improved checkpoint to {best_checkpoint_path} "
                f"(val accuracy: {best_val_accuracy:.4f})"
            )

        print(
            f"Epoch {current_epoch}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train acc: {train_metrics['accuracy']:.4f}, "
            f"train top 3 acc: {train_metrics['top_3_acc']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val acc: {val_metrics['accuracy']:.4f}, "
            f"val top 3 acc: {val_metrics['top_3_acc']:.4f}"
        )

    final_checkpoint_path = save_checkpoint(
        model,
        get_epoch_checkpoint_path(checkpoint_path, epochs),
    )
    print(f"Saved final-epoch checkpoint to {final_checkpoint_path}")
    elapsed_seconds = time.time() - start_time

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "best_val_accuracy": best_val_accuracy,
        "fine_tune": fine_tune,
        "backbone_learning_rate": (
            backbone_learning_rate if fine_tune else None
        ),
        "classifier_learning_rate": (
            classifier_learning_rate if fine_tune else learning_rate
        ),
        "num_unfrozen_layers": unfrozen_backbone_layers,
        "checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
    }

