import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "src" / "CUB_200_2011_cropped_square"
DEFAULT_CLASSIFIER_CHECKPOINT = REPO_ROOT / "checkpoints" / "ra_vit_classifier.pt"
DEFAULT_FINE_TUNED_CHECKPOINT = REPO_ROOT / "checkpoints" / "ra_vit_fine_tuned.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.primary_model.models import RA_ViT  # noqa: E402
from src.primary_model.train import (  # noqa: E402
    fine_tune_model,
    get_device,
    train_classifier,
)


def load_ra_vit_model(
    checkpoint_path,
    *,
    device,
    num_classes,
    classifier_hidden_dim,
    dropout,
    attention_layer_index,
    local_crop_size,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain a model state dictionary.")

    model = RA_ViT(
        num_classes=num_classes,
        fc1_dim=classifier_hidden_dim,
        dropout=dropout,
        freeze_backbones=True,
        attention_layer_index=attention_layer_index,
        local_crop_size=local_crop_size,
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint is incompatible with the requested RA-ViT architecture. "
            "Check --num-classes and --classifier-hidden-dim."
        ) from error

    model.to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune an RA-ViT bird classifier.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue classifier-head training from --checkpoint.",
    )
    mode.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune backbone blocks from --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="RA-ViT checkpoint required by --resume and --fine-tune.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        help="Base path for epoch checkpoints.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd"),
        default="adamw",
    )
    parser.add_argument("--loss", choices=("ce",), default="ce")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--attention-layer-index", type=int, default=-1)
    parser.add_argument("--local-crop-size", type=int, default=7)
    parser.add_argument("--num-unfrozen-blocks", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing epoch checkpoints to be replaced.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    if (args.resume or args.fine_tune) and args.checkpoint is None:
        parser.error("--checkpoint is required with --resume or --fine-tune.")
    if args.checkpoint is not None and not (args.resume or args.fine_tune):
        parser.error("--checkpoint requires --resume or --fine-tune.")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.data_root.is_dir():
        parser.error(f"dataset directory does not exist: {args.data_root}")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive.")
    if args.classifier_lr <= 0.0:
        parser.error("--classifier-lr must be positive.")
    if args.backbone_lr <= 0.0:
        parser.error("--backbone-lr must be positive.")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1).")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.num_classes < 2:
        parser.error("--num-classes must be at least 2.")
    class_count = sum(path.is_dir() for path in args.data_root.iterdir())
    if class_count != args.num_classes:
        parser.error(
            f"dataset contains {class_count} class directories, but "
            f"--num-classes is {args.num_classes}."
        )
    if args.classifier_hidden_dim < 1:
        parser.error("--classifier-hidden-dim must be at least 1.")
    if args.local_crop_size < 1:
        parser.error("--local-crop-size must be at least 1.")
    if not 1 <= args.num_unfrozen_blocks <= 12:
        parser.error("--num-unfrozen-blocks must be between 1 and 12.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable.")


def resolve_device(device_name):
    if device_name == "auto":
        return get_device()
    return torch.device(device_name)


def resolve_output_checkpoint(args):
    if args.output_checkpoint is not None:
        return args.output_checkpoint
    if args.fine_tune:
        return DEFAULT_FINE_TUNED_CHECKPOINT
    return DEFAULT_CLASSIFIER_CHECKPOINT


def ensure_output_is_available(checkpoint_path, epochs, overwrite):
    checkpoint_path = Path(checkpoint_path)
    existing_paths = [
        Path(f"{checkpoint_path}_e{epoch}")
        for epoch in range(1, epochs + 1)
        if Path(f"{checkpoint_path}_e{epoch}").exists()
    ]
    if existing_paths and not overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:3])
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint(s): {preview}. "
            "Choose --output-checkpoint or pass --overwrite."
        )


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    if args.checkpoint is not None:
        return load_ra_vit_model(
            args.checkpoint,
            device=device,
            num_classes=args.num_classes,
            classifier_hidden_dim=args.classifier_hidden_dim,
            dropout=args.dropout,
            attention_layer_index=args.attention_layer_index,
            local_crop_size=args.local_crop_size,
        )
    return RA_ViT(
        num_classes=args.num_classes,
        fc1_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        freeze_backbones=True,
        attention_layer_index=args.attention_layer_index,
        local_crop_size=args.local_crop_size,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_checkpoint = resolve_output_checkpoint(args)
    ensure_output_is_available(output_checkpoint, args.epochs, args.overwrite)
    set_random_seed(args.seed)

    model = build_model(args, device)
    common_args = {
        "model": model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "data_root": args.data_root,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": device,
        "checkpoint_path": output_checkpoint,
    }

    mode = "fine-tune" if args.fine_tune else "resume" if args.resume else "train"
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_root}")
    print(f"Checkpoint output base: {output_checkpoint}")

    if args.fine_tune:
        result = fine_tune_model(
            backbone_learning_rate=args.backbone_lr,
            classifier_learning_rate=args.classifier_lr,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            **common_args,
        )
    else:
        result = train_classifier(
            learning_rate=args.learning_rate,
            **common_args,
        )

    print(
        "Training complete. "
        f"Best checkpoint: {result['checkpoint_path']}. "
        f"Elapsed time: {result['elapsed_seconds']:.1f} seconds."
    )


if __name__ == "__main__":
    main()
