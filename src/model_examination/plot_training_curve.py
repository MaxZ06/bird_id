import re
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "curve_to_plot.txt"
OUTPUT_PATH = (
    REPO_ROOT / "produced_visuals" / "training_curves" / "primary_model" / "training_curves.png"
)
BASELINE_OUTPUT_PATH = (
    REPO_ROOT
    / "produced_visuals"
    / "training_curves"
    / "baseline_model"
    / "ViT_baseline_training_curves.png"
)

BASELINE_RESULTS = """
Epoch 1/10 | train loss: 3.4286, train acc: 0.2728, train top 3 acc: 0.4491, val loss: 2.1364, val acc: 0.5121, val top 3 acc: 0.7197
Epoch 2/10 | train loss: 1.6496, train acc: 0.6029, train top 3 acc: 0.8155, val loss: 1.6004, val acc: 0.5701, val top 3 acc: 0.7995
Epoch 3/10 | train loss: 1.0940, train acc: 0.7184, train top 3 acc: 0.8992, val loss: 1.4086, val acc: 0.6147, val top 3 acc: 0.8301
Epoch 4/10 | train loss: 0.8296, train acc: 0.7835, train top 3 acc: 0.9379, val loss: 1.3170, val acc: 0.6453, val top 3 acc: 0.8434
Epoch 5/10 | train loss: 0.6743, train acc: 0.8221, train top 3 acc: 0.9521, val loss: 1.3134, val acc: 0.6413, val top 3 acc: 0.8395
Epoch 6/10 | train loss: 0.5505, train acc: 0.8526, train top 3 acc: 0.9661, val loss: 1.2821, val acc: 0.6641, val top 3 acc: 0.8449
Epoch 7/10 | train loss: 0.4617, train acc: 0.8747, train top 3 acc: 0.9727, val loss: 1.2877, val acc: 0.6609, val top 3 acc: 0.8489
Epoch 8/10 | train loss: 0.4342, train acc: 0.8759, train top 3 acc: 0.9749, val loss: 1.3150, val acc: 0.6515, val top 3 acc: 0.8379
Epoch 9/10 | train loss: 0.3520, train acc: 0.9076, train top 3 acc: 0.9835, val loss: 1.2471, val acc: 0.6766, val top 3 acc: 0.8528
Epoch 10/10 | train loss: 0.3305, train acc: 0.9069, train top 3 acc: 0.9850, val loss: 1.2734, val acc: 0.6703, val top 3 acc: 0.8559
""".strip()

LINE_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+(?:\.\d+)?),\s+"
    r"train global acc:\s+(?P<train_global_acc>\d+(?:\.\d+)?),\s+"
    r"train local acc:\s+(?P<train_local_acc>\d+(?:\.\d+)?),\s+"
    r"val loss:\s+(?P<val_loss>\d+(?:\.\d+)?),\s+"
    r"val global acc:\s+(?P<val_global_acc>\d+(?:\.\d+)?),\s+"
    r"val local acc:\s+(?P<val_local_acc>\d+(?:\.\d+)?),\s+"
    r"train summed acc:\s+(?P<train_summed_acc>\d+(?:\.\d+)?),\s+"
    r"val summed acc:\s+(?P<val_summed_acc>\d+(?:\.\d+)?)"
)

BASELINE_LINE_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+(?:\.\d+)?),\s+"
    r"train acc:\s+(?P<train_acc>\d+(?:\.\d+)?),\s+"
    r"train top 3 acc:\s+(?P<train_top_3_acc>\d+(?:\.\d+)?),\s+"
    r"val loss:\s+(?P<val_loss>\d+(?:\.\d+)?),\s+"
    r"val acc:\s+(?P<val_acc>\d+(?:\.\d+)?),\s+"
    r"val top 3 acc:\s+(?P<val_top_3_acc>\d+(?:\.\d+)?)"
)


def parse_curve_file(input_path=INPUT_PATH):
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_global_acc": [],
        "val_global_acc": [],
        "train_local_acc": [],
        "val_local_acc": [],
        "train_summed_acc": [],
        "val_summed_acc": [],
    }

    for line_number, line in enumerate(Path(input_path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        match = LINE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Could not parse line {line_number}: {line}")

        for key in metrics:
            value = match.group(key)
            metrics[key].append(int(value) if key == "epoch" else float(value))

    if not metrics["epoch"]:
        raise ValueError(f"No curve data found in {input_path}")

    return metrics


def plot_training_curves(metrics, output_path=OUTPUT_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    epochs = metrics["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, metrics["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, metrics["val_loss"], marker="o", label="Val loss")
    axes[0].set_title("Train and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        metrics["train_global_acc"],
        marker="o",
        label="Train global acc",
    )
    axes[1].plot(
        epochs,
        metrics["train_local_acc"],
        marker="o",
        label="Train local acc",
    )
    axes[1].plot(
        epochs,
        metrics["train_summed_acc"],
        marker="o",
        label="Train summed acc",
    )
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        metrics["val_global_acc"],
        marker="o",
        label="Val global acc",
    )
    axes[2].plot(
        epochs,
        metrics["val_local_acc"],
        marker="o",
        label="Val local acc",
    )
    axes[2].plot(
        epochs,
        metrics["val_summed_acc"],
        marker="o",
        label="Val summed acc",
    )
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    print(f"Saved training curves to {output_path}")
    plt.show()


def parse_baseline_results(results=BASELINE_RESULTS):
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_top_3_acc": [],
        "val_top_3_acc": [],
    }

    for line_number, line in enumerate(results.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        match = BASELINE_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Could not parse baseline line {line_number}: {line}")

        for key in metrics:
            value = match.group(key)
            metrics[key].append(int(value) if key == "epoch" else float(value))

    return metrics


def plot_baseline(output_path=BASELINE_OUTPUT_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    metrics = parse_baseline_results()
    epochs = metrics["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, metrics["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, metrics["val_loss"], marker="o", label="Val loss")
    axes[0].set_title("Baseline Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, metrics["train_acc"], marker="o", label="Train acc")
    axes[1].set_title("Baseline Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, metrics["val_acc"], marker="o", label="Val acc")
    axes[2].set_title("Baseline Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    print(f"Saved baseline training curves to {output_path}")
    plt.show()


def main():
    metrics = parse_curve_file()
    plot_training_curves(metrics)


if __name__ == "__main__":
    main()
