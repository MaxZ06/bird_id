import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import create_vit_b16_dataloaders
from src.primary_model.models import RA_ViT, linear_combiner, weighted_logit_combiner




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

            global_logits, local_logits = model(images)
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


def calculate_epoch_metrics_combiner(
    model,
    combiner,
    dataloader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None
    model.eval()
    combiner.train() if is_training else combiner.eval()

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

            # get global and local outputs from the classifier
            with torch.no_grad():
                global_logits, local_logits = model(images)
                combined_logits = torch.cat((global_logits, local_logits), dim=1)

            total_logits = combiner(combined_logits)
            loss = criterion(total_logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (total_logits.argmax(dim=1) == labels).sum().item()
            top_3_predictions = total_logits.topk(3, dim=1).indices
            total_top_3_correct += (
                top_3_predictions == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "top_3_acc": total_top_3_correct / total_examples,
    }


def calculate_epoch_metrics_weighted_combiner(
    model,
    combiner,
    dataloader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None
    model.eval()
    combiner.train() if is_training else combiner.eval()

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

            with torch.no_grad():
                global_logits, local_logits = model(images)

            total_logits = combiner(global_logits, local_logits)
            loss = criterion(total_logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (total_logits.argmax(dim=1) == labels).sum().item()
            top_3_predictions = total_logits.topk(3, dim=1).indices
            total_top_3_correct += (
                top_3_predictions == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "top_3_acc": total_top_3_correct / total_examples,
    }

def generate_testing_log(bs, lr, epoch, optimizer, loss, history, for_combiner=False):
    path = Path(__file__).resolve().parents[2] / "testing_logs" / "logs_by_version.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() == False:
        with path.open("w") as gen_file:
            gen_file.write("testing log: Max Zhong\n")

    if for_combiner == False:
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
    
# will be called right after a generate test log for classifier
    if for_combiner == True:
        with path.open("a") as append_file:
            append_file.write("combiner statistic:\n")
            for e in range(epoch):
                train_metrics = history[e]['train']
                val_metrics = history[e]['val']
                append_file.write(
                f"Epoch {e + 1}/{epoch} | "
                f"train loss: {train_metrics['loss']:.4f}, "
                f"train acc: {train_metrics['accuracy']:.4f}, "
                f"train top 3 acc: {train_metrics['top_3_acc']:.4f}, "
                f"val loss: {val_metrics['loss']:.4f}, "
                f"val acc: {val_metrics['accuracy']:.4f}, "
                f"val top 3 acc: {val_metrics['top_3_acc']:.4f}\n"
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
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train global acc: {train_metrics['global_accuracy']:.4f}, "
            f"train local acc: {train_metrics['local_accuracy']:.4f}, "
            f"train summed acc: {train_metrics['summed_accuracy']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val global acc: {val_metrics['global_accuracy']:.4f}, "
            f"val local acc: {val_metrics['local_accuracy']:.4f}, "
            f"val summed acc: {val_metrics['summed_accuracy']:.4f} "
        )

    elapsed_seconds = time.time() - start_time
    generate_testing_log(bs=batch_size, lr=learning_rate, epoch=epochs,
                          optimizer=optimizer_name, loss=loss_name, history=history)
    checkpoint_path = save_checkpoint(model, checkpoint_path)

    return {
        "model": model,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": checkpoint_path,
    }


def train_linear_combiner(
    classifier_model,
    batch_size=32,
    learning_rate=0.001,
    epochs=5,
    optimizer="adam",
    criterion="ce",
    combiner=None,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path=Path(__file__).resolve().parents[2] / "checkpoints" / "linear_combiner.pt",
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

    classifier_model.to(device)
    classifier_model.eval()
    for parameter in classifier_model.parameters():
        parameter.requires_grad = False

    combiner = combiner or linear_combiner(
        summed_logits=2 * len(class_names),
        out_logits=len(class_names),
    )
    combiner.to(device)

    optimizer = get_optimizer(optimizer, combiner.parameters(), learning_rate)
    criterion = get_loss(criterion)

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = calculate_epoch_metrics_combiner(
            classifier_model,
            combiner,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = calculate_epoch_metrics_combiner(
            classifier_model,
            combiner,
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
    generate_testing_log(bs=batch_size, lr=learning_rate, epoch=epochs,
                          optimizer="adam", loss="ce", history=history, for_combiner=True)
    elapsed_seconds = time.time() - start_time
    checkpoint_path = save_checkpoint(combiner, checkpoint_path)

    return {
        "model": classifier_model,
        "combiner": combiner,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": checkpoint_path,
    }


def train_weighted_combiner(
    classifier_model,
    batch_size=32,
    learning_rate=0.001,
    epochs=5,
    optimizer="adam",
    criterion="ce",
    combiner=None,
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    checkpoint_path=Path(__file__).resolve().parents[2] / "checkpoints" / "weighted_combiner.pt",
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

    classifier_model.to(device)
    classifier_model.eval()
    for parameter in classifier_model.parameters():
        parameter.requires_grad = False

    combiner = combiner or weighted_logit_combiner()
    combiner.to(device)

    optimizer = get_optimizer(optimizer, combiner.parameters(), learning_rate)
    criterion = get_loss(criterion)

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = calculate_epoch_metrics_weighted_combiner(
            classifier_model,
            combiner,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = calculate_epoch_metrics_weighted_combiner(
            classifier_model,
            combiner,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        w1 = torch.sigmoid(combiner.raw_w1).item()
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train acc: {train_metrics['accuracy']:.4f}, "
            f"train top 3 acc: {train_metrics['top_3_acc']:.4f}, "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val acc: {val_metrics['accuracy']:.4f}, "
            f"val top 3 acc: {val_metrics['top_3_acc']:.4f}, "
            f"w1: {w1:.4f}"
        )
    generate_testing_log(bs=batch_size, lr=learning_rate, epoch=epochs,
                          optimizer="adam", loss="ce", history=history, for_combiner=True)
    elapsed_seconds = time.time() - start_time
    checkpoint_path = save_checkpoint(combiner, checkpoint_path)

    return {
        "model": classifier_model,
        "combiner": combiner,
        "history": history,
        "test_loader": test_loader,
        "class_names": class_names,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_path": checkpoint_path,

    }



if __name__ == "__main__":
    classifier_model = RA_ViT(num_classes=200, freeze_backbones=True)
    train_classifier(
        epochs=2,
        model=classifier_model,
        learning_rate=0.005,
        batch_size=128,
        checkpoint_path=Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "ra_vit_classifier_preprocessed_nblur_ngauss_10e_comb0.5_lr0.005bs128.pt",
    )
