from argparse import ArgumentParser
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SRC_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SRC_ROOT / "CUB_200_2011_cropped_square"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_size(image_path):
    with Image.open(image_path) as image:
        return image.size


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


def find_matching_images(data_root, max_size, max_laplacian_variance):
    matching_images = []

    for image_path in get_image_paths(data_root):
        width, height = get_image_size(image_path)
        if width >= max_size and height >= max_size:
            continue

        blur_score = compute_laplacian_blur_score(image_path)
        if blur_score >= max_laplacian_variance:
            continue

        matching_images.append({
            "path": image_path,
            "width": width,
            "height": height,
            "blur_score": blur_score,
        })

    return matching_images


def show_random_matching_image(data_root=DATA_ROOT, max_size=224, max_laplacian_variance=100.0):
    matching_images = find_matching_images(
        data_root,
        max_size=max_size,
        max_laplacian_variance=max_laplacian_variance,
    )

    if not matching_images:
        print(
            "No images matched: "
            f"size smaller than {max_size} and "
            f"Laplacian variance smaller than {max_laplacian_variance}."
        )
        return None

    sample = random.choice(matching_images)
    with Image.open(sample["path"]) as image:
        display_image = image.convert("RGB")
        plt.figure(figsize=(6, 6))
        plt.imshow(display_image)

    plt.axis("off")
    plt.title(
        f"{sample['path'].parent.name}/{sample['path'].name}\n"
        f"size: {sample['width']}x{sample['height']} | "
        f"Laplacian variance: {sample['blur_score']:.2f}\n"
        f"{len(matching_images)} matching images found",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()

    print(f"Displayed: {sample['path']}")
    print(f"Size: {sample['width']}x{sample['height']}")
    print(f"Laplacian variance: {sample['blur_score']:.2f}")
    print(f"Matching images found: {len(matching_images)}")
    return sample


def parse_args():
    parser = ArgumentParser(
        description=(
            "Randomly display an image whose size is smaller than x and "
            "Laplacian variance is smaller than y."
        ),
    )
    parser.add_argument(
        "x",
        type=int,
        help="Maximum image size threshold. Matches if width or height is smaller than x.",
    )
    parser.add_argument(
        "y",
        type=float,
        help="Maximum Laplacian variance threshold.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to the ImageFolder-style dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_random_matching_image(
        data_root=args.data_root,
        max_size=args.x,
        max_laplacian_variance=args.y,
    )


if __name__ == "__main__":
    main()
