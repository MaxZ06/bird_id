import argparse
import sys
import textwrap
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import (
    CROPPED_DATA_ROOT,
    vit_b16_eval_transform,
)
from src.primary_model.training_ground import load_ra_vit_model


CLASSIFIER_CHECKPOINT_PATH = (
    REPO_ROOT / "checkpoints" / "final_stage" / "RA_ViT_fine_tuned_v3_e24"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOP_K = 3
IMAGES_PER_FIGURE = 6


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names(data_root=CROPPED_DATA_ROOT):
    dataset = datasets.ImageFolder(root=data_root)
    return dataset.classes


def load_primary_model(
    classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    device=None,
):
    device = device or get_device()
    class_names = get_class_names()
    if len(class_names) != 200:
        raise ValueError(f"Expected 200 CUB classes, found {len(class_names)}.")

    classifier = load_ra_vit_model(
        checkpoint_path=classifier_checkpoint_path,
        device=device,
    )
    return classifier, class_names, device


def evaluate_cub_species_accuracy(
    classifier,
    data_root=CROPPED_DATA_ROOT,
    batch_size=32,
    num_workers=0,
    device=None,
):
    """Evaluate RA-ViT on all cropped CUB images and print per-class accuracy."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Could not find cropped CUB data: {data_root}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    device = device or next(classifier.parameters()).device
    dataset = datasets.ImageFolder(
        root=data_root,
        transform=vit_b16_eval_transform,
    )
    if len(dataset.classes) != 200:
        raise ValueError(
            f"Expected 200 CUB species folders, found {len(dataset.classes)}."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    class_correct = torch.zeros(len(dataset.classes), dtype=torch.long)
    class_total = torch.zeros(len(dataset.classes), dtype=torch.long)

    classifier.eval()
    print(
        f"Evaluating {len(dataset)} cropped CUB images across "
        f"{len(dataset.classes)} species..."
    )
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            probabilities = classifier(images)
            predictions = probabilities.argmax(dim=1).cpu()
            labels = labels.cpu()

            class_total += torch.bincount(
                labels,
                minlength=len(dataset.classes),
            )
            class_correct += torch.bincount(
                labels[predictions.eq(labels)],
                minlength=len(dataset.classes),
            )

    results = {}
    print("\nPer-species accuracy:")
    for class_index, class_name in enumerate(dataset.classes):
        correct = class_correct[class_index].item()
        total = class_total[class_index].item()
        accuracy = correct / total if total else 0.0
        results[class_name] = accuracy
        print(f"{class_name}: {accuracy:.2%} ({correct}/{total})")

    total_correct = class_correct.sum().item()
    total_images = class_total.sum().item()
    overall_accuracy = total_correct / total_images
    print(
        f"\nOverall accuracy: {overall_accuracy:.2%} "
        f"({total_correct}/{total_images})"
    )
    return results


def select_image_folder():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select an image folder")
    root.destroy()
    return Path(folder) if folder else None


def find_images(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def top_predictions(probabilities, class_names, top_k=TOP_K):
    top_probabilities, top_indices = probabilities.topk(top_k, dim=1)
    return [
        {
            "class_name": class_names[class_index],
            "probability": probability,
        }
        for class_index, probability in zip(
            top_indices[0].cpu().tolist(),
            top_probabilities[0].cpu().tolist(),
        )
    ]


def predict_images(
    classifier,
    image_paths,
    class_names,
    device,
    top_k=TOP_K,
):
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    predictions = []

    with torch.inference_mode():
        for image_path in image_paths:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            final_probabilities = classifier(image_tensor)

            predictions.append({
                "image": image,
                "path": image_path,
                "top_predictions": top_predictions(
                    final_probabilities,
                    class_names,
                    top_k=top_k,
                ),
            })

    return predictions


def display_predictions(predictions, images_per_figure=IMAGES_PER_FIGURE):
    for start in range(0, len(predictions), images_per_figure):
        page = predictions[start:start + images_per_figure]
        columns = min(3, len(page))
        rows = (len(page) + columns - 1) // columns
        displayed_top_k = len(page[0]["top_predictions"])
        cell_height = 5.5 + 0.35 * displayed_top_k
        figure = plt.figure(
            figsize=(6 * columns, cell_height * rows),
            constrained_layout=True,
        )
        page_grid = figure.add_gridspec(rows, columns)

        for index, prediction in enumerate(page):
            row, column = divmod(index, columns)
            text_height = max(1.2, 0.35 * (displayed_top_k + 1))
            cell_grid = page_grid[row, column].subgridspec(
                2,
                1,
                height_ratios=(4.5, text_height),
                hspace=0.04,
            )
            image_axis = figure.add_subplot(cell_grid[0])
            text_axis = figure.add_subplot(cell_grid[1])

            prediction_text = "\n".join(
                f"{rank}. {item['class_name'].split('.', 1)[-1].replace('_', ' ')}: "
                f"{item['probability']:.2%}"
                for rank, item in enumerate(prediction["top_predictions"], 1)
            )
            image_axis.imshow(prediction["image"])
            image_axis.axis("off")
            image_axis.set_title(
                textwrap.fill(prediction["path"].name, width=34),
                fontsize=10,
                pad=8,
            )
            text_axis.axis("off")
            text_axis.text(
                0.02,
                0.98,
                prediction_text,
                transform=text_axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                linespacing=1.35,
            )

        for index in range(len(page), rows * columns):
            row, column = divmod(index, columns)
            empty_axis = figure.add_subplot(page_grid[row, column])
            empty_axis.axis("off")

        figure.suptitle(
            f"Images {start + 1}-{start + len(page)} of {len(predictions)}",
            fontsize=13,
        )
        plt.show()


def examine_folder(
    classifier,
    class_names,
    device,
    folder,
    top_k=TOP_K,
    images_per_figure=IMAGES_PER_FIGURE,
):
    image_paths = find_images(folder)
    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(image_paths)} images. Producing predictions...")
    predictions = predict_images(
        classifier,
        image_paths,
        class_names,
        device,
        top_k=top_k,
    )

    for prediction in predictions:
        print(prediction["path"])
        for rank, item in enumerate(prediction["top_predictions"], 1):
            print(
                f"  {rank}. {item['class_name']}: "
                f"{item['probability']:.2%}"
            )

    display_predictions(
        predictions,
        images_per_figure=images_per_figure,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display RA-ViT predictions for an image folder."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="RA-ViT checkpoint to load instead of the default.",
    )
    parser.add_argument(
        "--evaluate-cub-species",
        action="store_true",
        help=(
            "Evaluate RA-ViT on every cropped CUB image and print accuracy "
            "for each species."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for full CUB evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for full CUB evaluation (default: 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of predictions to display per image (default: 3).",
    )
    parser.add_argument(
        "--images-per-page",
        type=int,
        default=IMAGES_PER_FIGURE,
        help="Maximum images in each figure window (default: 6).",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 200:
        parser.error("--top-k must be between 1 and 200.")
    if args.images_per_page < 1:
        parser.error("--images-per-page must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    return args


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or CLASSIFIER_CHECKPOINT_PATH
    classifier, class_names, device = load_primary_model(
        classifier_checkpoint_path=checkpoint_path,
    )

    print(f"Loaded RA-ViT: {checkpoint_path}")
    print(f"Using device: {device}")

    if args.evaluate_cub_species:
        evaluate_cub_species_accuracy(
            classifier,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        return

    while True:
        folder = select_image_folder()
        if folder is None:
            break

        print(f"\nSelected folder: {folder}")
        examine_folder(
            classifier,
            class_names,
            device,
            folder,
            top_k=args.top_k,
            images_per_figure=args.images_per_page,
        )

        keep_going = input("Select another folder? [y/N]: ").strip().lower()
        if keep_going != "y":
            break


if __name__ == "__main__":
    main()
