from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from data_splitting import create_vit_b16_dataloaders


DEFAULT_LOG_PATH = Path("testing_log.txt")
DEFAULT_CHECKPOINT_DIR = Path("fc_checkpoints")
NUM_BIRD_CLASSES = 200
VIT_B16_FEATURE_DIM = 768


class ViT_B_16_baseline(nn.Module):
    def __init__(
        self,
        num_classes=NUM_BIRD_CLASSES,
        hidden_dim=512,
        dropout=0.3,
        freeze_backbone=True,
    ):
        super(ViT_B_16_baseline, self).__init__()
        self.name = "ViT_B_16_baseline"
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(VIT_B16_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_epoch_metrics(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
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
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    error = 1.0 - accuracy
    return average_loss, accuracy, error


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device=None,
    after_epoch_callback=None,
):
    """General training loop for any model that maps images to class logits."""
    if device is None:
        device = get_device()

    model = model.to(device)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_error": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_error": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy, train_error = calculate_epoch_metrics(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy, val_error = calculate_epoch_metrics(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["train_error"].append(train_error)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_error"].append(val_error)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}, "
            f"val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}"
        )

        if after_epoch_callback is not None:
            after_epoch_callback(epoch, model, history)

    return model, history


def write_testing_log(log_path, run_parameters, history):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Testing Log\n")
        log_file.write(f"created_at: {datetime.now().isoformat(timespec='seconds')}\n\n")

        log_file.write("Run Parameters\n")
        for key, value in run_parameters.items():
            log_file.write(f"{key}: {value}\n")

        log_file.write("\nEpoch Metrics\n")
        for epoch_index in range(len(history["train_loss"])):
            log_file.write(
                f"{epoch_index + 1}. train loss: "
                f"{history['train_loss'][epoch_index]:.6f}, train accuracy: "
                f"{history['train_accuracy'][epoch_index]:.6f}, train error: "
                f"{history['train_error'][epoch_index]:.6f}, val loss: "
                f"{history['val_loss'][epoch_index]:.6f}, val accuracy: "
                f"{history['val_accuracy'][epoch_index]:.6f},val error: "
                f"{history['val_error'][epoch_index]:.6f}\n"
            )


def save_fc_checkpoint(model, checkpoint_dir, epoch):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fc_layers_epoch_{epoch:03d}.pt"
    torch.save(model.classifier.state_dict(), checkpoint_path)
    print(f"Saved FC checkpoint to {checkpoint_path}")


def build_trainable_optimizer(model, learning_rate, weight_decay):
    trainable_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    return torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001
    hidden_dim = 32
    dropout = 0.3
    freeze_backbone = True
    checkpoint_every = 2
    seed = 1
    device = get_device()

    train_loader, val_loader, test_loader, class_names = create_vit_b16_dataloaders(
        batch_size=batch_size,
        seed=seed,
    )
    model = ViT_B_16_baseline(
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_trainable_optimizer(model, learning_rate, weight_decay)

    def after_epoch(epoch, model, history):
        if checkpoint_every and epoch % checkpoint_every == 0:
            save_fc_checkpoint(model, DEFAULT_CHECKPOINT_DIR, epoch)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        after_epoch_callback=after_epoch,
    )

    run_parameters = {
        "model": model.name,
        "num_classes": len(class_names),
        "trained_layers": "classifier only",
        "loss": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "regularization": f"dropout={dropout}, weight_decay={weight_decay}",
        "num_epochs": num_epochs,
        "device": device,
        "freeze_backbone": freeze_backbone,
        "classifier_hidden_dim": hidden_dim,
        "classifier_dropout": dropout,
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        "checkpoint_every": checkpoint_every,
        "checkpoint_contents": "model.classifier.state_dict() only",
        "train_examples": len(train_loader.dataset),
        "val_examples": len(val_loader.dataset),
        "seed": seed,
    }
    write_testing_log(DEFAULT_LOG_PATH, run_parameters, history)
    print(f"Saved testing log to {DEFAULT_LOG_PATH}")
