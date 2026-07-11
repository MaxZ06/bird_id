from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


# Clean data before train/val/test splitting by removing cropped images that are
# too small to preserve useful fine-grained details.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "CUB_200_2011_cropped_square"
MIN_WIDTH = 200
MIN_HEIGHT = 200
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
        description="Remove cropped CUB images smaller than 200x200 pixels.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    removed_paths = remove_small_images(args.data_root, apply=args.apply)
    class_counts = count_images_by_class(args.data_root)

    action = "Removed" if args.apply else "Would remove"
    print(f"{action} {len(removed_paths)} images smaller than {MIN_WIDTH}x{MIN_HEIGHT}.")
    if not args.apply:
        print("Preview only. Run with --apply to delete these images.")

    total_images = sum(class_counts.values())
    print(f"Total images after cleaning: {total_images}")
    print_class_distribution(class_counts)


if __name__ == "__main__":
    main()

