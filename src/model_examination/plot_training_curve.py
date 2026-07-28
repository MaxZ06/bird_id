import re
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "curve_to_plot.txt"
OUTPUT_PATH = (
    REPO_ROOT
    / "produced_visuals"
    / "training_curves"
    / "primary_model"
    / "training_curves.png"
)

EPOCH_PATTERN = re.compile(
    r"Epoch\s+\d+/\d+\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+\.\d+),\s+"
    r"train global acc:\s+(?P<train_global_acc>\d+\.\d+),\s+"
    r"train local acc:\s+(?P<train_local_acc>\d+\.\d+),\s+"
    r"val loss:\s+(?P<val_loss>\d+\.\d+),\s+"
    r"val global acc:\s+(?P<val_global_acc>\d+\.\d+),\s+"
    r"val local acc:\s+(?P<val_local_acc>\d+\.\d+),\s+"
    r"train summed logit acc:\s+(?P<train_summed_acc>\d+\.\d+),\s+"
    r"val summed logit acc:\s+(?P<val_summed_acc>\d+\.\d+)"
)


def parse_training_log(input_path=INPUT_PATH):
    metric_names = EPOCH_PATTERN.groupindex
    metrics = {"epoch": []}
    metrics.update({name: [] for name in metric_names})
    ignore_block = False

    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "//":
            ignore_block = not ignore_block
            continue
        if ignore_block:
            continue

        match = EPOCH_PATTERN.fullmatch(line)
        if match is None:
            continue

        metrics["epoch"].append(len(metrics["epoch"]) + 1)
        for name in metric_names:
            metrics[name].append(float(match.group(name)))

    if not metrics["epoch"]:
        raise ValueError(f"No usable epoch records found in {input_path}")

    return metrics


def plot_pair(axis, epochs, train_values, val_values, title, ylabel):
    axis.plot(epochs, train_values, label="Train", linewidth=2)
    axis.plot(epochs, val_values, label="Validation", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Continuous epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()


def plot_training_curves(metrics, output_path=OUTPUT_PATH):
    epochs = metrics["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    plot_pair(
        axes[0, 0],
        epochs,
        metrics["train_loss"],
        metrics["val_loss"],
        "Loss",
        "Cross-entropy loss",
    )
    plot_pair(
        axes[0, 1],
        epochs,
        metrics["train_global_acc"],
        metrics["val_global_acc"],
        "Global Accuracy",
        "Accuracy",
    )
    plot_pair(
        axes[1, 0],
        epochs,
        metrics["train_local_acc"],
        metrics["val_local_acc"],
        "Local Accuracy",
        "Accuracy",
    )
    plot_pair(
        axes[1, 1],
        epochs,
        metrics["train_summed_acc"],
        metrics["val_summed_acc"],
        "Summed-Logit Accuracy",
        "Accuracy",
    )

    for axis in axes.flat[1:]:
        axis.set_ylim(0, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plotted {len(epochs)} continuous epochs")
    print(f"Saved training curves to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_curves(parse_training_log())
