"""Square-crop the extracted CUB/NABirds overlap using NABirds bounding boxes."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

from PIL import Image

from data_preprocessing import crop_square_around_box


def read_bounding_boxes(zf: zipfile.ZipFile) -> dict[str, tuple[float, float, float, float]]:
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with zf.open("bounding_boxes.txt") as stream:
        for raw in stream:
            image_id, x, y, width, height = raw.decode("utf-8").split()
            boxes[image_id] = tuple(float(value) for value in (x, y, width, height))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--input", type=Path, default=Path("NAbirds"))
    parser.add_argument("--output", type=Path, default=Path("NAbirds_cropped_square"))
    args = parser.parse_args()

    manifest_path = args.input / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    with zipfile.ZipFile(args.archive) as zf:
        boxes = read_bounding_boxes(zf)

    args.output.mkdir(parents=True, exist_ok=True)
    output_metadata = args.output / "processed_images.csv"
    metadata_fields = [
        "image_id",
        "cub_id",
        "cub_class",
        "nabirds_species",
        "nabirds_visual_category",
        "source_path",
        "processed_path",
        "source_width",
        "source_height",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "square_x",
        "square_y",
        "square_side",
    ]

    with output_metadata.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=metadata_fields)
        writer.writeheader()

        for index, row in enumerate(rows, 1):
            image_id = row["image_id"]
            bbox = boxes.get(image_id)
            if bbox is None:
                raise KeyError(f"No NABirds bounding box for image {image_id}")

            source_path = args.input / Path(row["output_path"])
            target_path = args.output / Path(row["output_path"])
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source_path) as image:
                source_width, source_height = image.size
                cropped, (square_x, square_y, square_side) = crop_square_around_box(image, bbox)
                cropped.save(target_path, quality=95)

            writer.writerow(
                {
                    "image_id": image_id,
                    "cub_id": row["cub_id"],
                    "cub_class": row["cub_class"],
                    "nabirds_species": row["nabirds_species"],
                    "nabirds_visual_category": row["nabirds_visual_category"],
                    "source_path": source_path.as_posix(),
                    "processed_path": target_path.as_posix(),
                    "source_width": source_width,
                    "source_height": source_height,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_width": bbox[2],
                    "bbox_height": bbox[3],
                    "square_x": square_x,
                    "square_y": square_y,
                    "square_side": square_side,
                }
            )

            if index % 1000 == 0:
                print(f"Cropped {index}/{len(rows)} images", flush=True)

    print(f"Complete: {len(rows)} square crops saved to {args.output}")
    print(f"Crop metadata saved to {output_metadata}")


if __name__ == "__main__":
    main()
