from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Clean data before train/val/test splitting by removing cropped images that are
# too small to preserve useful fine-grained details.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATA_ROOT = SRC_ROOT / "CUB_200_2011_cropped_square"
VISUALS_ROOT = REPO_ROOT / "produced_visuals" / "data_set_distributions"
BLUR_DISTRIBUTION_PLOT = VISUALS_ROOT / "blur_distribution.png"
BLUR_SAMPLE_PLOT = VISUALS_ROOT / "blur_sample_images.png"
MIN_WIDTH = 112
MIN_HEIGHT = 112
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(data_root):
    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_smaller_than_threshold(image_path, min_width=MIN_WIDTH, min_height=MIN_HEIGHT):
    with Image.open(image_path) as image:
        width, height = image.size
    return width < min_width or height < min_height


def compute_laplacian_blur_score(image_path):
    with Image.open(image_path) as image:
        gray_image = image.convert("L")
        pixels = np.asarray(gray_image, dtype=np.float32)

    laplacian = (
        -4 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(laplacian.var())


def compute_blur_scores(data_root=DATA_ROOT):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    return {
        image_path: compute_laplacian_blur_score(image_path)
        for image_path in get_image_paths(data_root)
    }


def remove_blurry_images(data_root=DATA_ROOT, min_laplacian_score=100.0, apply=False):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    removed_paths = []
    for image_path in get_image_paths(data_root):
        blur_score = compute_laplacian_blur_score(image_path)
        if blur_score < min_laplacian_score:
            removed_paths.append((image_path, blur_score))
            if apply:
                image_path.unlink()

    return removed_paths


def plot_blur_distribution(
    blur_scores,
    output_path=BLUR_DISTRIBUTION_PLOT,
    bins=50,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.hist(list(blur_scores.values()), bins=bins)
    plt.xlabel("Blurriness Level (Laplacian Variance)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Image Blurriness")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def sample_blur_images_by_level(blur_scores, levels=5, samples_per_level=4):
    sorted_scores = sorted(blur_scores.items(), key=lambda item: item[1])
    if not sorted_scores:
        return []

    level_samples = []
    for level_index in range(levels):
        start = level_index * len(sorted_scores) // levels
        end = (level_index + 1) * len(sorted_scores) // levels
        level_items = sorted_scores[start:end]
        if not level_items:
            continue

        sample_indices = np.linspace(
            0,
            len(level_items) - 1,
            num=min(samples_per_level, len(level_items)),
            dtype=int,
        )
        level_samples.append([
            level_items[sample_index]
            for sample_index in sample_indices
        ])

    return level_samples


def plot_blur_sample_images(
    level_samples,
    output_path=BLUR_SAMPLE_PLOT,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(level_samples)
    cols = max((len(samples) for samples in level_samples), default=0)
    if rows == 0 or cols == 0:
        raise ValueError("No blur samples to plot.")

    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for row_index, samples in enumerate(level_samples):
        scores = [score for _, score in samples]
        for col_index in range(cols):
            axis = axes[row_index, col_index]
            axis.axis("off")
            if col_index >= len(samples):
                continue

            image_path, score = samples[col_index]
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))

            if col_index == 0:
                axis.set_ylabel(
                    f"Level {row_index + 1}\n{min(scores):.1f}-{max(scores):.1f}",
                    rotation=0,
                    labelpad=55,
                    va="center",
                )
            axis.set_title(f"{score:.1f}", fontsize=9)

    figure.suptitle(
        "Sample Images by Laplacian Blur Level\nLower scores are blurrier; higher scores are sharper",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def count_images_by_class(data_root):
    class_counts = {}
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        class_counts[class_dir.name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return class_counts


def remove_small_images(data_root=DATA_ROOT, apply=False):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    removed_paths = []
    for image_path in get_image_paths(data_root):
        if is_smaller_than_threshold(image_path):
            removed_paths.append(image_path)
            if apply:
                image_path.unlink()

    return removed_paths


def print_class_distribution(class_counts):
    print("New distribution of number of images vs bird class:")
    for class_name, image_count in class_counts.items():
        print(f"{class_name}: {image_count}")


def parse_args():
    parser = ArgumentParser(
        description="Remove cropped CUB images smaller than 224x224 pixels.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to the ImageFolder-style cropped CUB dataset.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the small images. Without this flag, only preview.",
    )
    parser.add_argument(
        "--plot-blur",
        action="store_true",
        help="Plot the dataset blurriness distribution using Laplacian variance.",
    )
    parser.add_argument(
        "--blur-output",
        type=Path,
        default=BLUR_DISTRIBUTION_PLOT,
        help="Path to save the Laplacian blur distribution plot.",
    )
    parser.add_argument(
        "--blur-bins",
        type=int,
        default=50,
        help="Number of histogram bins for the blur distribution plot.",
    )
    parser.add_argument(
        "--plot-blur-samples",
        action="store_true",
        help="Plot sample images across Laplacian blur-score levels.",
    )
    parser.add_argument(
        "--blur-sample-output",
        type=Path,
        default=BLUR_SAMPLE_PLOT,
        help="Path to save the sampled blur-level image plot.",
    )
    parser.add_argument(
        "--blur-levels",
        type=int,
        default=5,
        help="Number of blur levels to sample from.",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=4,
        help="Number of images to show for each blur level.",
    )
    parser.add_argument(
        "--remove-blurry-below",
        type=float,
        default=None,
        help="Remove images with Laplacian variance below this score. Requires --apply to delete.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    removed_paths = remove_small_images(args.data_root, apply=args.apply)
    class_counts = count_images_by_class(args.data_root)

    action = "Removed" if args.apply else "Would remove"
    print(f"{action} {len(removed_paths)} images smaller than {MIN_WIDTH}x{MIN_HEIGHT}.")
    if not args.apply:
        print("Preview only. Run with --apply to delete these images.")

    if args.remove_blurry_below is not None:
        blurry_paths = remove_blurry_images(
            args.data_root,
            min_laplacian_score=args.remove_blurry_below,
            apply=args.apply,
        )
        print(
            f"{action} {len(blurry_paths)} images with Laplacian variance "
            f"below {args.remove_blurry_below}."
        )

    total_images = sum(class_counts.values())
    print(f"Total images after cleaning: {total_images}")
    print_class_distribution(class_counts)

    if args.plot_blur:
        blur_scores = compute_blur_scores(args.data_root)
        output_path = plot_blur_distribution(
            blur_scores,
            output_path=args.blur_output,
            bins=args.blur_bins,
        )
        print(f"Saved blur distribution plot to {output_path}")

    if args.plot_blur_samples:
        blur_scores = compute_blur_scores(args.data_root)
        level_samples = sample_blur_images_by_level(
            blur_scores,
            levels=args.blur_levels,
            samples_per_level=args.samples_per_level,
        )
        output_path = plot_blur_sample_images(
            level_samples,
            output_path=args.blur_sample_output,
        )
        print(f"Saved blur sample image plot to {output_path}")


if __name__ == "__main__":
    main()
