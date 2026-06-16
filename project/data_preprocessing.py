from pathlib import Path
import csv
import math
import tarfile

from PIL import Image


ARCHIVE_PATH = Path("CUB_200_2011.tgz")
DATA_ROOT = Path("CUB_200_2011")
OUTPUT_ROOT = Path("CUB_200_2011_cropped_square")
METADATA_PATH = OUTPUT_ROOT / "processed_images.csv"


def extract_cub_if_needed(archive_path=ARCHIVE_PATH, data_root=DATA_ROOT):
    if data_root.exists():
        return
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing {archive_path}. Put CUB_200_2011.tgz in this folder.")

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(path=Path.cwd(), filter="data")
        except TypeError:
            tar.extractall(path=Path.cwd())


def read_image_paths(data_root=DATA_ROOT):
    image_paths = {}
    with (data_root / "images.txt").open("r", encoding="utf-8") as f:
        for line in f:
            image_id, rel_path = line.strip().split(maxsplit=1)
            image_paths[int(image_id)] = rel_path
    return image_paths


def read_class_labels(data_root=DATA_ROOT):
    labels = {}
    with (data_root / "image_class_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            image_id, class_id = line.strip().split()
            labels[int(image_id)] = int(class_id)
    return labels


def read_class_names(data_root=DATA_ROOT):
    class_names = {}
    with (data_root / "classes.txt").open("r", encoding="utf-8") as f:
        for line in f:
            class_id, class_name = line.strip().split(maxsplit=1)
            class_names[int(class_id)] = class_name
    return class_names


def read_bounding_boxes(data_root=DATA_ROOT):
    boxes = {}
    with (data_root / "bounding_boxes.txt").open("r", encoding="utf-8") as f:
        for line in f:
            image_id, x, y, width, height = line.strip().split()
            boxes[int(image_id)] = tuple(float(v) for v in (x, y, width, height))
    return boxes


def crop_square_around_box(image, bbox):
    x, y, box_width, box_height = bbox
    box_left = math.floor(x)
    box_top = math.floor(y)
    box_right = math.ceil(x + box_width)
    box_bottom = math.ceil(y + box_height)

    side = max(box_right - box_left, box_bottom - box_top)
    center_x = x + box_width / 2
    center_y = y + box_height / 2

    square_left = math.floor(center_x - side / 2)
    square_top = math.floor(center_y - side / 2)

    # Preserve the labeled box fully if integer rounding shifts the square by one pixel.
    if square_left > box_left:
        square_left = box_left
    if square_top > box_top:
        square_top = box_top
    if square_left + side < box_right:
        square_left = box_right - side
    if square_top + side < box_bottom:
        square_top = box_bottom - side

    square_right = square_left + side
    square_bottom = square_top + side

    image = image.convert("RGB")
    image_width, image_height = image.size
    crop_left = max(square_left, 0)
    crop_top = max(square_top, 0)
    crop_right = min(square_right, image_width)
    crop_bottom = min(square_bottom, image_height)

    cropped_square = Image.new("RGB", (side, side), (0, 0, 0))
    visible_region = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    paste_x = crop_left - square_left
    paste_y = crop_top - square_top
    cropped_square.paste(visible_region, (paste_x, paste_y))

    return cropped_square, (square_left, square_top, side)


def preprocess_cub_images(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, overwrite=True):
    extract_cub_if_needed(data_root=data_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = read_image_paths(data_root)
    class_labels = read_class_labels(data_root)
    class_names = read_class_names(data_root)
    bounding_boxes = read_bounding_boxes(data_root)

    rows = []
    for index, image_id in enumerate(sorted(image_paths), start=1):
        rel_path = image_paths[image_id]
        class_id = class_labels[image_id]
        species = class_names[class_id]
        source_path = data_root / "images" / rel_path
        target_dir = output_root / species
        target_path = target_dir / Path(rel_path).name

        target_dir.mkdir(parents=True, exist_ok=True)
        bbox = bounding_boxes[image_id]
        with Image.open(source_path) as image:
            cropped_square, (square_x, square_y, square_side) = crop_square_around_box(image, bbox)
            if overwrite or not target_path.exists():
                cropped_square.save(target_path, quality=95)

        rows.append({
            "image_id": image_id,
            "class_id": class_id,
            "species": species,
            "source_path": str(source_path),
            "processed_path": str(target_path),
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_width": bbox[2],
            "bbox_height": bbox[3],
            "square_x": square_x,
            "square_y": square_y,
            "square_side": square_side,
        })

        if index % 500 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} cropped images to {output_root}")
    print(f"Saved labels and crop metadata to {METADATA_PATH}")


preprocess_cub_images()