from pathlib import Path
import statistics

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "CUB_200_2011_cropped_square"
CLASS_COUNT_PLOT = PROJECT_ROOT / "project\data_set_distributions\class_image_counts.png"
IMAGE_SIZE_PLOT = PROJECT_ROOT / "project\data_set_distributions\image_size_distribution.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images_by_class(data_root=DATA_ROOT):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    class_counts = {}
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        image_count = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_counts[class_dir.name] = image_count

    return class_counts


def get_image_paths(data_root=DATA_ROOT):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")

    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images_by_size_range(data_root=DATA_ROOT, bin_size=50):
    size_counts = {}
    for image_path in get_image_paths(data_root):
        with Image.open(image_path) as image:
            width, height = image.size

        size = max(width, height)
        lower = (size // bin_size) * bin_size
        upper = lower + bin_size
        label = f"{lower}x{lower}-{upper}x{upper}"
        size_counts[label] = size_counts.get(label, 0) + 1

    return dict(
        sorted(
            size_counts.items(),
            key=lambda item: int(item[0].split("x", maxsplit=1)[0]),
        )
    )


def get_detailed_stats(class_counts):
    l = 100
    h = 0
    for b_name, num_img in class_counts.items():
        if num_img < l:
            l = num_img
            lname = b_name
        if num_img > h:
            h = num_img
            hname = b_name
    return lname, hname


def get_class_count_distribution_stats(class_counts, sample_std=False):
    image_counts = list(class_counts.values())
    if not image_counts:
        raise ValueError("class_counts must contain at least one class.")

    std_function = statistics.stdev if sample_std else statistics.pstdev
    return {
        "median": statistics.median(image_counts),
        "std": std_function(image_counts),
    }


def plot_class_counts(class_counts, output_path=CLASS_COUNT_PLOT):
    class_names = list(class_counts.keys())
    image_counts = list(class_counts.values())

    plt.figure(figsize=(32, 8))
    plt.bar(class_names, image_counts)
    plt.xlabel("Bird Species")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Bird Species")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    plt.ylim(0, 61)
    return output_path


def plot_size_distribution(size_counts, output_path=IMAGE_SIZE_PLOT):
    size_ranges = list(size_counts.keys())
    image_counts = list(size_counts.values())

    plt.figure(figsize=(16, 8))
    plt.bar(size_ranges, image_counts)
    plt.xlabel("Image Size Range")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Image Sizes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main():
    class_counts = count_images_by_class()
    size_counts = count_images_by_size_range()
    total_images = sum(class_counts.values())

    print(f"Dataset root: {DATA_ROOT}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Total images: {total_images}")

    print(f"avg img per class: {total_images / 200}")
    distribution_stats = get_class_count_distribution_stats(class_counts)
    print(f"median img per class: {distribution_stats['median']}")
    print(f"std dev img per class: {distribution_stats['std']:.4f}")
 
    lname, hname = get_detailed_stats(class_counts)
    print(f"{hname} has the most img at {class_counts[hname]},{lname} has the least img at {class_counts[lname]} ")
    output_path = plot_class_counts(class_counts, output_path="project\data_set_distributions\class_image_counts_aftercleaning.png")
    print(f"Saved class count plot to {output_path}")
    size_plot_path = plot_size_distribution(size_counts, output_path="project\data_set_distributions\image_size_distribution_afterclean.png")
    print(f"Saved image size distribution plot to {size_plot_path}")

if __name__ == "__main__":
    main()
    
