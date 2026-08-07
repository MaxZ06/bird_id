import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "simple_resnet50.pt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model.baseline_model_resnet import (
    SimpleResNet50,
    get_device,
    train_simple_resnet50,
)


def _continued_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix or ".pt"
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_continued{suffix}"
    )


def load_and_train_saved_simple_resnet50(
    saved_checkpoint_path,
    output_checkpoint_path=None,
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
    data_root=None,
    num_workers=0,
    seed=42,
    device=None,
    dropout=0.3,
):
    """Load a saved SimpleResNet50 state dict and continue training it.

    The project's SimpleResNet50 checkpoints contain model weights only, so a
    new optimizer is created for the continued training run. The hidden and
    output dimensions are inferred from the saved classifier weights.
    """
    saved_checkpoint_path = Path(saved_checkpoint_path)
    if not saved_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Could not find SimpleResNet50 checkpoint: {saved_checkpoint_path}"
        )

    device = device or get_device()
    state_dict = torch.load(
        saved_checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Also accept a conventional checkpoint dictionary if one is supplied.
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    hidden_weight_key = "classifier.0.weight"
    output_weight_key = "classifier.4.weight"
    if hidden_weight_key not in state_dict or output_weight_key not in state_dict:
        raise ValueError(
            "The checkpoint is not a compatible SimpleResNet50 state dict: "
            f"missing {hidden_weight_key!r} or {output_weight_key!r}."
        )

    hidden_dim = state_dict[hidden_weight_key].shape[0]
    num_classes = state_dict[output_weight_key].shape[0]
    model = SimpleResNet50(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        weights=None,
    )
    model.load_state_dict(state_dict)

    if output_checkpoint_path is None:
        output_checkpoint_path = _continued_checkpoint_path(saved_checkpoint_path)

    result = train_simple_resnet50(
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune=fine_tune,
        backbone_learning_rate=backbone_learning_rate,
        classifier_learning_rate=classifier_learning_rate,
        num_unfrozen_layers=num_unfrozen_layers,
        weight_decay=weight_decay,
        epochs=epochs,
        optimizer=optimizer,
        loss=loss,
        model=model,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed,
        device=device,
        checkpoint_path=output_checkpoint_path,
    )
    result["loaded_checkpoint_path"] = saved_checkpoint_path
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a new SimpleResNet50 or continue training a saved checkpoint."
        )
    )
    parser.add_argument(
        "--saved-checkpoint",
        type=Path,
        default=None,
        help="Saved SimpleResNet50 weights to continue training.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help=(
            "Base output checkpoint path. Improvement checkpoints receive an "
            "epoch suffix."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze the final ResNet50 residual blocks.",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=None,
        help="Classifier LR during fine-tuning; defaults to --learning-rate.",
    )
    parser.add_argument(
        "--num-unfrozen-layers",
        type=int,
        default=1,
        help="Number of final ResNet50 residual blocks to unfreeze (1-16).",
    )
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    common_training_args = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fine_tune": args.fine_tune,
        "backbone_learning_rate": args.backbone_lr,
        "classifier_learning_rate": args.classifier_lr,
        "num_unfrozen_layers": args.num_unfrozen_layers,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "data_root": args.data_root,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    if args.saved_checkpoint is not None:
        result = load_and_train_saved_simple_resnet50(
            saved_checkpoint_path=args.saved_checkpoint,
            output_checkpoint_path=args.output_checkpoint,
            dropout=args.dropout,
            **common_training_args,
        )
    else:
        output_checkpoint = args.output_checkpoint or DEFAULT_CHECKPOINT_PATH
        model = SimpleResNet50(dropout=args.dropout)
        result = train_simple_resnet50(
            model=model,
            checkpoint_path=output_checkpoint,
            **common_training_args,
        )

    print(
        "Training complete. "
        f"Best validation accuracy: {result['best_val_accuracy']:.4f}. "
        f"Best checkpoint: {result['checkpoint_path']}. "
        f"Final checkpoint: {result['final_checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
