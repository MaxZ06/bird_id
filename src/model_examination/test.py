import inspect
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.primary_model.train import get_device, create_vit_b16_dataloaders
from src.primary_model.training_ground import load_ra_vit_model
from src.baseline_model.baseline_model_resnet import SimpleResNet50
from src.data_preprocessing.data_splitting import vit_b16_eval_transform


SUBJECT_TEST_ROOT = REPO_ROOT / "bird_id_test_subject" / "test_images"
RESNET50_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "simple_resnet50_e32.pt"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def load_resnet_bmodel(
    checkpoint_path=RESNET50_CHECKPOINT_PATH,
    device=None,
):
    """Load a saved SimpleResNet50 model and prepare it for testing."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Could not find SimpleResNet50 checkpoint: {checkpoint_path}"
        )

    device = device or get_device()
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    hidden_weight_key = "classifier.0.weight"
    output_weight_key = "classifier.4.weight"
    if hidden_weight_key not in state_dict or output_weight_key not in state_dict:
        raise ValueError(
            "The checkpoint is not a compatible SimpleResNet50 state dict: "
            f"missing {hidden_weight_key!r} or {output_weight_key!r}."
        )

    model = SimpleResNet50(
        num_classes=state_dict[output_weight_key].shape[0],
        hidden_dim=state_dict[hidden_weight_key].shape[0],
        weights=None,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_baseline(model, dataloader, device=None):
    """Evaluate a baseline classifier and print only its top-1 accuracy."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

    if total_examples == 0:
        raise ValueError("Cannot evaluate accuracy on an empty dataloader.")

    accuracy = total_correct / total_examples
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy



def evaluate_accuracy(model, dataloader, mode="sum", combiner=None, device=None):
    """Evaluate global, local, and final classification accuracy."""
    if mode not in {"sum", "combiner"}:
        raise ValueError("mode must be either 'sum' or 'combiner'.")
    if mode == "combiner" and combiner is None:
        raise ValueError("combiner is required when mode='combiner'.")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if combiner is not None:
        combiner.eval()
        combiner_input_count = len(
            inspect.signature(combiner.forward).parameters
        )
        if combiner_input_count not in {1, 2}:
            raise ValueError(
                "combiner.forward must accept either one concatenated-logits "
                "argument or separate global and local logits."
            )

    global_correct = 0
    local_correct = 0
    final_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            global_logits, local_logits = model(
                images,
                return_branch_logits=True,
            )

            if mode == "combiner":
                if combiner_input_count == 1:
                    combined_logits = torch.cat(
                        (global_logits, local_logits),
                        dim=1,
                    )
                    final_logits = combiner(combined_logits)
                else:
                    final_logits = combiner(global_logits, local_logits)
            else:
                final_logits = global_logits + local_logits

            global_correct += (
                global_logits.argmax(dim=1) == labels
            ).sum().item()
            local_correct += (
                local_logits.argmax(dim=1) == labels
            ).sum().item()
            final_correct += (
                final_logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += labels.size(0)

    if total_examples == 0:
        raise ValueError("Cannot evaluate accuracy on an empty dataloader.")

    accuracies = {
        "global_accuracy": global_correct / total_examples,
        "local_accuracy": local_correct / total_examples,
        "final_accuracy": final_correct / total_examples,
    }

    print(f"Global accuracy: {accuracies['global_accuracy']:.4f}")
    print(f"Local accuracy: {accuracies['local_accuracy']:.4f}")
    print(f"Final accuracy: {accuracies['final_accuracy']:.4f}")

    return accuracies


class LabeledImageDataset(Dataset):
    """Images whose labels are listed one-per-line in filename order."""

    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.targets[index]


def create_subject_test_loader(
    class_names,
    data_root=SUBJECT_TEST_ROOT,
    batch_size=32,
    num_workers=0,
):
    """Load test images and their one-per-line CUB labels from labels.txt."""
    data_root = Path(data_root)
    labels_path = data_root / "labels.txt"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Could not find test image folder: {data_root}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Could not find test labels: {labels_path}")

    image_paths = sorted(
        (
            path
            for path in data_root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    label_names = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(image_paths) != len(label_names):
        raise ValueError(
            f"Found {len(image_paths)} images but {len(label_names)} labels in "
            f"{labels_path}. Labels must be listed in sorted filename order."
        )

    targets = []
    for image_path, label_name in zip(image_paths, label_names):
        try:
            cub_index = int(label_name.split(".", maxsplit=1)[0]) - 1
        except ValueError as error:
            raise ValueError(
                f"Label for {image_path.name} does not start with a CUB class ID: "
                f"{label_name!r}"
            ) from error

        if not 0 <= cub_index < len(class_names):
            raise ValueError(
                f"Invalid CUB class ID for {image_path.name}: {label_name!r}"
            )
        if class_names[cub_index] != label_name:
            raise ValueError(
                f"Label mismatch for {image_path.name}: {label_name!r} maps to "
                f"{class_names[cub_index]!r}."
            )
        targets.append(cub_index)

    dataset = LabeledImageDataset(
        image_paths=image_paths,
        targets=targets,
        transform=vit_b16_eval_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )



if __name__ == "__main__":
    device = get_device()
    # get datas
    dataloader_kwargs = {
        "batch_size": 32,
        "num_workers": 0,
        "seed": 42,
        }

    train_loader, val_loader, test_loader, class_names = create_vit_b16_dataloaders(
        **dataloader_kwargs,
    )

    subject_test_loader = create_subject_test_loader(
        class_names=class_names,
        batch_size=dataloader_kwargs["batch_size"],
        num_workers=dataloader_kwargs["num_workers"],
    )
    print(
        f"Loaded {len(subject_test_loader.dataset)} labeled test images "
        f"from {SUBJECT_TEST_ROOT}."
    )


    # weighted_combiner = initialize_weighted_combiner(0.5, device)
    # load current best classifier

    ra_vit_model = load_ra_vit_model(
        checkpoint_path=REPO_ROOT
        / "checkpoints"
        / "final_stage"
        / "RA_ViT_fine_tuned_v3_e24",
        device=device,
    )

    evaluate_accuracy(
        model=ra_vit_model,
        dataloader=subject_test_loader,
        mode="sum",
        device=device,
    )





"""
    baseline_model = load_resnet_bmodel(
        checkpoint_path=REPO_ROOT
        / "checkpoints"
        / "simple_resnet50_e32_continued_e29.pt",
        device=device
    )
    evaluate_baseline(
        model=baseline_model,
        dataloader=subject_test_loader,
        device=device
    )
"""

