"""Extract CUB-200 species shared with NABirds from the NABirds ZIP archive."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path


ALIASES = {
    "brewer blackbird": "Brewer's Blackbird",
    "cardinal": "Northern Cardinal",
    "chuck will widow": "Chuck-will's-widow",
    "brandt cormorant": "Brandt's Cormorant",
    "heermann gull": "Heermann's Gull",
    "anna hummingbird": "Anna's Hummingbird",
    "florida jay": "Florida Scrub-Jay",
    "mockingbird": "Northern Mockingbird",
    "nighthawk": "Common Nighthawk",
    "clark nutcracker": "Clark's Nutcracker",
    "scott oriole": "Scott's Oriole",
    "white pelican": "American White Pelican",
    "sayornis": "Say's Phoebe",
    "geococcyx": "Greater Roadrunner",
    "great grey shrike": "Northern Shrike",
    "baird sparrow": "Baird's Sparrow",
    "brewer sparrow": "Brewer's Sparrow",
    "harris sparrow": "Harris's Sparrow",
    "henslow sparrow": "Henslow's Sparrow",
    "le conte sparrow": "Le Conte's Sparrow",
    "lincoln sparrow": "Lincoln's Sparrow",
    "nelson sharp tailed sparrow": "Nelson's Sparrow",
    "tree sparrow": "American Tree Sparrow",
    "artic tern": "Arctic Tern",
    "myrtle warbler": "Yellow-rumped Warbler",
    "swainson warbler": "Swainson's Warbler",
    "wilson warbler": "Wilson's Warbler",
    "bewick wren": "Bewick's Wren",
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("’", "").replace("'", "")).strip()


def read_table(zf: zipfile.ZipFile, name: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with zf.open(name) as stream:
        for raw in stream:
            key, value = raw.decode("utf-8").rstrip("\r\n").split(" ", 1)
            rows[key] = value
    return rows


def safe_folder(cub_id: str, cub_name: str) -> str:
    return f"{cub_id}.{cub_name.replace(' ', '_')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--cub-classes", type=Path, default=Path("CUB_200_2011/classes.txt"))
    parser.add_argument("--output", type=Path, default=Path("NAbirds"))
    args = parser.parse_args()

    cub_rows: list[tuple[str, str, str]] = []
    for line in args.cub_classes.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\d+\s+(\d{3})\.(.+)", line)
        if match:
            cub_id, raw_name = match.groups()
            cub_name = raw_name.replace("_", " ")
            target_name = ALIASES.get(normalize(cub_name), cub_name)
            cub_rows.append((cub_id, cub_name, target_name))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    summary_path = args.output / "species_summary.csv"

    with zipfile.ZipFile(args.archive) as zf:
        classes = read_table(zf, "classes.txt")
        parents = read_table(zf, "hierarchy.txt")
        image_labels = read_table(zf, "image_class_labels.txt")
        image_paths = read_table(zf, "images.txt")
        class_by_name = {normalize(name): class_id for class_id, name in classes.items()}

        selected: dict[str, tuple[str, str, str]] = {}
        for cub_id, cub_name, target_name in cub_rows:
            class_id = class_by_name.get(normalize(target_name))
            if class_id is not None:
                selected[class_id] = (cub_id, cub_name, target_name)

        if len(selected) != 142:
            raise RuntimeError(f"Expected 142 matched species, found {len(selected)}")

        owner_cache: dict[str, tuple[str, str, str] | None] = {}

        def selected_owner(class_id: str) -> tuple[str, str, str] | None:
            if class_id in owner_cache:
                return owner_cache[class_id]
            current = class_id
            seen: set[str] = set()
            while current not in seen:
                seen.add(current)
                if current in selected:
                    owner_cache[class_id] = selected[current]
                    return selected[current]
                current = parents.get(current, "")
                if not current:
                    break
            owner_cache[class_id] = None
            return None

        selected_images: list[tuple[str, str, str, str, str, str]] = []
        for image_id, leaf_id in image_labels.items():
            owner = selected_owner(leaf_id)
            if owner is None:
                continue
            cub_id, cub_name, target_name = owner
            selected_images.append(
                (image_id, image_paths[image_id], leaf_id, classes[leaf_id], cub_id, cub_name)
            )

        counts: Counter[str] = Counter()
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(
                ["cub_id", "cub_class", "nabirds_species", "nabirds_visual_category", "image_id", "source_path", "output_path"]
            )
            for index, (image_id, source_rel, leaf_id, visual_name, cub_id, cub_name) in enumerate(selected_images, 1):
                folder = safe_folder(cub_id, cub_name)
                output_rel = Path(folder) / Path(source_rel).name
                destination = args.output / output_rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open("images/" + source_rel) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                target_name = ALIASES.get(normalize(cub_name), cub_name)
                writer.writerow([cub_id, cub_name, target_name, visual_name, image_id, source_rel, output_rel.as_posix()])
                counts[cub_id] += 1
                if index % 1000 == 0:
                    print(f"Extracted {index}/{len(selected_images)} images", flush=True)

        with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(["cub_id", "cub_class", "nabirds_species", "image_count"])
            for cub_id, cub_name, target_name in sorted(selected.values()):
                writer.writerow([cub_id, cub_name, target_name, counts[cub_id]])

    empty = [cub_id for cub_id, _, _ in selected.values() if counts[cub_id] == 0]
    if empty:
        raise RuntimeError(f"Matched taxonomy nodes without images: {empty}")
    print(f"Complete: {len(selected)} species and {sum(counts.values())} images extracted to {args.output}")


if __name__ == "__main__":
    main()
