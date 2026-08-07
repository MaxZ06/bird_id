import re
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import create_vit_b16_dataloaders
from src.primary_model.models import RA_ViT




def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_feedforward_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.global_classifier.parameters():
        parameter.requires_grad = True

    for parameter in model.local_classifier.parameters():
        parameter.requires_grad = True


def set_fine_tuning_trainable(model, num_unfrozen_blocks=2):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for classifier in (model.global_classifier, model.local_classifier):
        for parameter in classifier.parameters():
            parameter.requires_grad = True

    for backbone in (model.global_vit, model.local_vit):
        encoder_layers = list(backbone.encoder.layers.children())
        if not 1 <= num_unfrozen_blocks <= len(encoder_layers):
            raise ValueError(
                "num_unfrozen_blocks must be between 1 and "
                f"{len(encoder_layers)}."
            )

        for layer in encoder_layers[-num_unfrozen_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        for parameter in backbone.encoder.ln.parameters():
            parameter.requires_grad = True


def get_optimizer(optimizer, parameters, learning_rate, weight_decay=0.0):
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



# function to run training or evaluation for one epoch of data
def calculate_epoch_metrics_classifier(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_global_correct = 0
    total_local_correct = 0
    total_summed_correct = 0
    total_examples = 0


    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            # current loss: separate loss functions
            global_logits, local_logits = model(
                images,
                return_branch_logits=True,
            )
            total_logits = global_logits + local_logits
            loss = (
                criterion(global_logits, labels)
                + criterion(local_logits, labels)
            )


            if is_training:
                loss.backward()
                optimizer.step()            

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_global_correct += (global_logits.argmax(dim=1) == labels).sum().item()
            total_local_correct += (local_logits.argmax(dim=1) == labels).sum().item()
            total_summed_correct += (total_logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "global_accuracy": total_global_correct / total_examples,
        "local_accuracy": total_local_correct / total_examples,
        "summed_accuracy": total_summed_correct / total_examples,
    }

def generate_testing_log(
    bs,
    lr,
    epoch,
    optimizer,
    loss,
    history,
    dropout=None,
    weight_decay=None,
    training_mode="backbone frozen",
):
    path = (
        Path(__file__).resolve().parents[2]
        / "testing_logs"
        / "classifier_training.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w") as gen_file:
            gen_file.write("classifier training log:\n")

    lines = path.read_text().splitlines()
    versions = [line for line in lines if line.startswith("version")]
    version_numbers = []
    for version in versions:
        match = re.match(r"version\s+(\d+)", version)
        if match is not None:
            version_numbers.append(int(match.group(1)))
    last_ver = max(version_numbers, default=0)

    with path.open("a") as w_file:
        w_file.write(f"version {last_ver + 1}:\n")
        w_file.write(
            f"RA_ViT with bs = {bs}, lr = {lr}, num trained epoch = {epoch}, "
            f"optimizer = {optimizer}, loss function = {loss}\n"
        )
        w_file.write(
            f"crop size = 7, {training_mode}, "
            "fc layer in classifiers = 786 -> 512 -> 200\n"
        )
        w_file.write(
            f"with dropout = {dropout}, weight decay = {weight_decay}, "
            "batch normalization applied\n\n"
        )

        for e in range(epoch):
            train_metrics = history[e]["train"]
            val_metrics = history[e]["val"]
            w_file.write(
                f"Epoch {e + 1}/{epoch} | "
                f"train loss: {train_metrics['loss']:.4f}, "
                f"train global acc: {train_metrics['global_accuracy']:.4f}, "
                f"train local acc: {train_metrics['local_accuracy']:.4f}, "
                f"val loss: {val_metrics['loss']:.4f}, "
                f"val global acc: {val_metrics['global_accuracy']:.4f}, "
                f"val local acc: {val_metrics['local_accuracy']:.4f}, "
                f"train summed logit acc: {train_metrics['summed_accuracy']:.4f}, "
                f"val summed logit acc: {val_metrics['summed_accuracy']:.4f}\n"
            )


def train_classifier(
    batch_size=32,
    learning_rate=0.001,
    epochs=5,
    optimizer="adam",
    loss="ce",
    dropout=0.3,
    weight_decay=0.0,
    model=None,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path=Path(__file__).resolve().parents[2] / "checkpoints" / "ra_vit_classifier.pt",
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

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0.0, 1.0).")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")

    model = model or RA_ViT(
        num_classes=len(class_names),
        dropout=dropout,
        freeze_backbones=True,
    )
    for classifier in (model.global_classifier, model.local_classifier):
        for module in classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    model.to(device)
    set_feedforward_trainable(model)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_name = optimizer
    loss_name = loss
    optimizer = get_optimizer(
        optimizer,
        trainable_parameters,
        learning_rate,
        weight_decay=weight_decay,
    )
    criterion = get_loss(loss)

    history = []
    best_val_accuracy = float("-inf")
    best_checkpoint_path = None
    start_time = time.time()

    for epoch in range(epochs):
        current_epoch = epoch + 1
        train_metrics = calculate_epoch_metrics_classifier(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = calculate_epoch_metrics_classifier(
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

        if val_metrics["summed_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["summed_accuracy"]
            best_checkpoint_path = save_checkpoint(
                model,
                f"{checkpoint_path}_e{current_epoch}",
            )

        print(
            f"Epoch {current_epoch}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train global acc: {train_metrics['global_accuracy']:.4f}, "
            f"train local acc: {train_metrics['local_accuracy']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val global acc: {val_metrics['global_accuracy']:.4f}, "
            f"val local acc: {val_metrics['local_accuracy']:.4f}, "
            f"train summed acc: {train_metrics['summed_accuracy']:.4f}, "
            f"val summed acc: {val_metrics['summed_accuracy']:.4f} "
        )

    elapsed_seconds = time.time() - start_time
    generate_testing_log(
        bs=batch_size,
        lr=learning_rate,
        epoch=epochs,
        optimizer=optimizer_name,
        loss=loss_name,
        history=history,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": best_checkpoint_path,
    }


def fine_tune_model(
    model,
    batch_size=32,
    backbone_learning_rate=1e-5,
    classifier_learning_rate=1e-4,
    epochs=5,
    num_unfrozen_blocks=2,
    optimizer="adamw",
    loss="ce",
    dropout=0.3,
    weight_decay=1e-4,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path=Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "ra_vit_fine_tuned.pt",
):
    if backbone_learning_rate <= 0.0:
        raise ValueError("backbone_learning_rate must be positive.")
    if classifier_learning_rate <= 0.0:
        raise ValueError("classifier_learning_rate must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0.0, 1.0).")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")

    device = device or get_device()
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }
    if data_root is not None:
        dataloader_kwargs["data_root"] = data_root

    train_loader, val_loader, test_loader, class_names = (
        create_vit_b16_dataloaders(**dataloader_kwargs)
    )

    for classifier in (model.global_classifier, model.local_classifier):
        for module in classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    model.to(device)
    set_fine_tuning_trainable(
        model,
        num_unfrozen_blocks=num_unfrozen_blocks,
    )

    classifier_parameters = [
        parameter
        for classifier in (model.global_classifier, model.local_classifier)
        for parameter in classifier.parameters()
        if parameter.requires_grad
    ]
    backbone_parameters = [
        parameter
        for backbone in (model.global_vit, model.local_vit)
        for parameter in backbone.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = [
        {
            "params": classifier_parameters,
            "lr": classifier_learning_rate,
        },
        {
            "params": backbone_parameters,
            "lr": backbone_learning_rate,
        },
    ]

    optimizer_name = optimizer
    loss_name = loss
    optimizer = get_optimizer(
        optimizer,
        parameter_groups,
        classifier_learning_rate,
        weight_decay=weight_decay,
    )
    criterion = get_loss(loss)

    history = []
    best_val_accuracy = float("-inf")
    best_checkpoint_path = None
    start_time = time.time()

    for epoch in range(epochs):
        current_epoch = epoch + 1
        train_metrics = calculate_epoch_metrics_classifier(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = calculate_epoch_metrics_classifier(
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

        if val_metrics["summed_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["summed_accuracy"]
            best_checkpoint_path = save_checkpoint(
                model,
                f"{checkpoint_path}_e{current_epoch}",
            )

        print(
            f"Epoch {current_epoch}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train global acc: {train_metrics['global_accuracy']:.4f}, "
            f"train local acc: {train_metrics['local_accuracy']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val global acc: {val_metrics['global_accuracy']:.4f}, "
            f"val local acc: {val_metrics['local_accuracy']:.4f}, "
            f"train summed acc: {train_metrics['summed_accuracy']:.4f}, "
            f"val summed acc: {val_metrics['summed_accuracy']:.4f}"
        )

    elapsed_seconds = time.time() - start_time
    generate_testing_log(
        bs=batch_size,
        lr=(
            f"classifier {classifier_learning_rate}, "
            f"backbone {backbone_learning_rate}"
        ),
        epoch=epochs,
        optimizer=optimizer_name,
        loss=loss_name,
        history=history,
        dropout=dropout,
        weight_decay=weight_decay,
        training_mode=(
            f"last {num_unfrozen_blocks} transformer blocks unfrozen "
            "in each backbone"
        ),
    )

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": best_checkpoint_path,
        "backbone_learning_rate": backbone_learning_rate,
        "classifier_learning_rate": classifier_learning_rate,
        "num_unfrozen_blocks": num_unfrozen_blocks,
    }
