import re
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 15,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "curve_to_plot.txt"
OUTPUT_PATH = (
    REPO_ROOT
    / "produced_visuals"
    / "training_curves"
    / "model_comparison"
    / "training_curves.png"
)

PRIMARY_EPOCH_PATTERN = re.compile(
    r"Epoch\s+\d+/\d+\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+\.\d+),\s+"
    r"train global acc:\s+(?P<train_global_acc>\d+\.\d+),\s+"
    r"train local acc:\s+(?P<train_local_acc>\d+\.\d+),\s+"
    r"val loss:\s+(?P<val_loss>\d+\.\d+),\s+"
    r"val global acc:\s+(?P<val_global_acc>\d+\.\d+),\s+"
    r"val local acc:\s+(?P<val_local_acc>\d+\.\d+),\s+"
    r"train summed logit acc:\s+(?P<train_final_acc>\d+\.\d+),\s+"
    r"val summed logit acc:\s+(?P<val_final_acc>\d+\.\d+)"
)

BASELINE_EPOCH_PATTERN = re.compile(
    r"Epoch\s+\d+/\d+\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+\.\d+),\s+"
    r"train acc:\s+(?P<train_acc>\d+\.\d+),\s+"
    r"train top 3 acc:\s+\d+\.\d+,\s+"
    r"val loss:\s+(?P<val_loss>\d+\.\d+),\s+"
    r"val acc:\s+(?P<val_acc>\d+\.\d+),\s+"
    r"val top 3 acc:\s+\d+\.\d+"
)


def empty_metrics(pattern):
    metrics = {"epoch": []}
    metrics.update({name: [] for name in pattern.groupindex})
    return metrics


def append_epoch(metrics, pattern, line):
    match = pattern.fullmatch(line)
    if match is None:
        return False

    metrics["epoch"].append(len(metrics["epoch"]) + 1)
    for name in pattern.groupindex:
        metrics[name].append(float(match.group(name)))
    return True


def parse_training_log(input_path=INPUT_PATH):
    primary_metrics = empty_metrics(PRIMARY_EPOCH_PATTERN)
    baseline_metrics = empty_metrics(BASELINE_EPOCH_PATTERN)
    current_section = "primary"

    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("baseline"):
            current_section = "baseline"
            continue

        if current_section == "primary":
            append_epoch(primary_metrics, PRIMARY_EPOCH_PATTERN, line)
        else:
            append_epoch(baseline_metrics, BASELINE_EPOCH_PATTERN, line)

    if not primary_metrics["epoch"]:
        raise ValueError(f"No primary model epoch records found in {input_path}")
    if not baseline_metrics["epoch"]:
        raise ValueError(f"No baseline epoch records found in {input_path}")

    return {
        "primary": primary_metrics,
        "baseline": baseline_metrics,
    }


def set_axis_style(axis, title, ylabel, accuracy=False):
    axis.set_title(title, pad=14)
    axis.set_xlabel("Continuous epoch", labelpad=10)
    axis.set_ylabel(ylabel, labelpad=4)
    if accuracy:
        axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)


def plot_loss(axis, primary, baseline):
    axis.plot(
        baseline["epoch"],
        baseline["train_loss"],
        label="Baseline train loss",
        linewidth=2,
    )
    axis.plot(
        baseline["epoch"],
        baseline["val_loss"],
        label="Baseline val loss",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["train_loss"],
        label="Final model train loss",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["val_loss"],
        label="Final model val loss",
        linewidth=2,
    )
    set_axis_style(axis, "Loss", "Cross-entropy loss")
    axis.legend(loc="upper right", frameon=True)


def plot_train_accuracy(axis, primary, baseline):
    axis.plot(
        baseline["epoch"],
        baseline["train_acc"],
        label="Baseline train acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["train_final_acc"],
        label="Final model final acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["train_local_acc"],
        label="Final model local acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["train_global_acc"],
        label="Final model global acc",
        linewidth=2,
    )
    set_axis_style(axis, "Training Accuracy", "Top-1 accuracy", accuracy=True)
    axis.legend(loc="lower right", frameon=True)


def plot_val_accuracy(axis, primary, baseline):
    axis.plot(
        baseline["epoch"],
        baseline["val_acc"],
        label="Baseline val acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["val_final_acc"],
        label="Final model final val acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["val_local_acc"],
        label="Final model local val acc",
        linewidth=2,
    )
    axis.plot(
        primary["epoch"],
        primary["val_global_acc"],
        label="Final model global val acc",
        linewidth=2,
    )
    set_axis_style(axis, "Validation Accuracy", "Top-1 accuracy", accuracy=True)
    axis.legend(loc="lower right", frameon=True)


def save_figure(figure, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.1, wspace=0.28)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def plot_training_curves(metrics, output_path=OUTPUT_PATH):
    primary = metrics["primary"]
    baseline = metrics["baseline"]

    figure, axes = plt.subplots(1, 3, figsize=(30, 8), sharex=True)

    plot_loss(axes[0], primary, baseline)
    plot_train_accuracy(axes[1], primary, baseline)
    plot_val_accuracy(axes[2], primary, baseline)

    output_path = save_figure(figure, output_path)

    print(f"Plotted {len(primary['epoch'])} final model epochs")
    print(f"Plotted {len(baseline['epoch'])} baseline epochs")
    print(f"Saved training curves to {output_path}")
    plt.show()
    return output_path


if __name__ == "__main__":
    plot_training_curves(parse_training_log())
