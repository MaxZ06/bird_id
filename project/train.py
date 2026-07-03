import time
import torch
from torch import nn
from data_splitting import create_vit_b16_dataloaders
from models import RA_ViT
from pathlib import Path




def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_feedforward_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.global_classifier.parameters():
        parameter.requires_grad = True

    for parameter in model.local_classifier.parameters():
        parameter.requires_grad = True


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



# function to run training or evaluation for one epoch of data
def calculate_epoch_metrics(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_global_correct = 0
    total_local_correct = 0
    total_examples = 0


    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            global_logits, local_logits = model(images)
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
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "global_accuracy": total_global_correct / total_examples,
        "local_accuracy": total_local_correct / total_examples,
    }

def generate_testing_log(bs, lr, epoch, optimizer, loss, history):
    path = Path("testing_logs/logs_by_version.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() == False:
        with path.open("w") as gen_file:
            gen_file.write("testing log: Max Zhong\n")

    lines = path.read_text().splitlines()
    versions = [line for line in lines if line.startswith("version")]
    if versions:
        last_ver = int(versions[-1].split()[1].rstrip(":"))
    else:
        last_ver = 0
    with path.open("a") as w_file:
        w_file.write(f"version {last_ver + 1}:\n")
        w_file.write(f"RA_ViT with bs = {bs}, lr = {lr}, num trained epoch = {epoch}, optimizer = {optimizer}, loss function = {loss}\n")
        w_file.write("with dropout = 0.3, batch normalization applied\n\n")
        
        for e in range(epoch):
            train_metrics = history[e]['train']
            val_metrics = history[e]['val']
            w_file.write(
            f"Epoch {e + 1}/{epoch} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train global acc: {train_metrics['global_accuracy']:.4f}, "
            f"train local acc: {train_metrics['local_accuracy']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val global acc: {val_metrics['global_accuracy']:.4f}, "
            f"val local acc: {val_metrics['local_accuracy']:.4f}\n"
            )


def train_classifier(
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

    model = model or RA_ViT(num_classes=len(class_names), freeze_backbones=True)
    model.to(device)
    set_feedforward_trainable(model)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_name = optimizer
    loss_name = loss
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
            f"train global acc: {train_metrics['global_accuracy']:.4f}, "
            f"train local acc: {train_metrics['local_accuracy']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val global acc: {val_metrics['global_accuracy']:.4f}, "
            f"val local acc: {val_metrics['local_accuracy']:.4f}"
        )

    elapsed_seconds = time.time() - start_time
    generate_testing_log(bs=batch_size, lr=learning_rate, epoch=epochs,
                          optimizer=optimizer_name, loss=loss_name, history=history)

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
    }


if __name__ == "__main__":
    train_classifier(epochs=5)
    
