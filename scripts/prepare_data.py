import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_PATH = REPO_ROOT / "CUB_200_2011.tgz"
DEFAULT_RAW_DATA_ROOT = REPO_ROOT / "CUB_200_2011"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "CUB_200_2011_cropped_square"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_cleaning import (  # noqa: E402
    IMAGE_EXTENSIONS,
    MIN_HEIGHT,
    MIN_WIDTH,
    count_images_by_class,
    remove_blurry_images,
    remove_small_images,
)
from src.data_preprocessing.data_preprocessing import (  # noqa: E402
    extract_cub_if_needed,
    preprocess_cub_images,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract, crop, clean, and validate the CUB-200-2011 dataset. "
            "Cleaning is preview-only unless --apply-cleaning is supplied."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to CUB_200_2011.tgz.",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=DEFAULT_RAW_DATA_ROOT,
        help="Directory containing the extracted CUB metadata and images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for square-cropped ImageFolder data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace cropped images that already exist.",
    )
    parser.add_argument(
        "--skip-size-cleaning",
        action="store_true",
        help="Skip the minimum image-size scan.",
    )
    parser.add_argument(
        "--remove-blurry-below",
        type=float,
        default=None,
        metavar="SCORE",
        help="Select images below this Laplacian-variance score.",
    )
    parser.add_argument(
        "--apply-cleaning",
        action="store_true",
        help="Delete images selected by the cleaning scans.",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=200,
        help="Expected number of output class directories.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output dataset without preprocessing it.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    if args.expected_classes < 1:
        parser.error("--expected-classes must be at least 1.")
    if args.remove_blurry_below is not None and args.remove_blurry_below < 0.0:
        parser.error("--remove-blurry-below cannot be negative.")
    if args.validate_only:
        if not args.output_root.is_dir():
            parser.error(f"output dataset does not exist: {args.output_root}")
        if args.apply_cleaning:
            parser.error("--apply-cleaning cannot be used with --validate-only.")
        return
    if not args.raw_data_root.is_dir() and not args.archive.is_file():
        parser.error(
            "raw dataset is absent and the archive does not exist: "
            f"{args.archive}"
        )


def validate_dataset(data_root, expected_classes):
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Processed dataset does not exist: {data_root}")

    class_counts = count_images_by_class(data_root)
    if len(class_counts) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} class directories, found "
            f"{len(class_counts)} in {data_root}."
        )

    empty_classes = [name for name, count in class_counts.items() if count == 0]
    if empty_classes:
        preview = ", ".join(empty_classes[:5])
        raise ValueError(f"Found empty class directories: {preview}")

    total_images = sum(class_counts.values())
    if total_images == 0:
        raise ValueError(f"No supported images found in {data_root}.")

    unsupported_files = [
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and path.name != "processed_images.csv"
        and path.suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if unsupported_files:
        preview = ", ".join(str(path) for path in unsupported_files[:5])
        raise ValueError(f"Found unsupported files in the dataset: {preview}")

    return class_counts, total_images


def run_cleaning(args):
    action = "Removed" if args.apply_cleaning else "Would remove"

    if not args.skip_size_cleaning:
        small_images = remove_small_images(
            args.output_root,
            apply=args.apply_cleaning,
        )
        print(
            f"{action} {len(small_images)} images smaller than "
            f"{MIN_WIDTH}x{MIN_HEIGHT}."
        )

    if args.remove_blurry_below is not None:
        blurry_images = remove_blurry_images(
            args.output_root,
            min_laplacian_score=args.remove_blurry_below,
            apply=args.apply_cleaning,
        )
        print(
            f"{action} {len(blurry_images)} images with Laplacian variance "
            f"below {args.remove_blurry_below:g}."
        )

    if not args.apply_cleaning and (
        not args.skip_size_cleaning or args.remove_blurry_below is not None
    ):
        print("Cleaning preview only; pass --apply-cleaning to delete selected images.")


def main():
    args = parse_args()

    if not args.validate_only:
        print(f"Archive: {args.archive}")
        print(f"Raw dataset: {args.raw_data_root}")
        print(f"Processed dataset: {args.output_root}")
        extract_cub_if_needed(
            archive_path=args.archive,
            data_root=args.raw_data_root,
        )
        preprocess_cub_images(
            data_root=args.raw_data_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        run_cleaning(args)

    class_counts, total_images = validate_dataset(
        args.output_root,
        expected_classes=args.expected_classes,
    )
    print(
        f"Validation passed: {total_images} images across "
        f"{len(class_counts)} classes."
    )


if __name__ == "__main__":
    main()
