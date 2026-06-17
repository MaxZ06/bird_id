from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

from data_splitting import CROPPED_DATA_ROOT, vit_b16_transform
from RA_ViT_model import RA_ViT


RA_VIT_CHECKPOINT_DIRS = [
    Path("RA_ViT_model_checkpoints"),
    Path("ViT_RA_model_checkpts"),
]
RA_VIT_CHECKPOINT_NAME = "RA_ViT_bs_32.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names(data_root=CROPPED_DATA_ROOT):
    dataset = datasets.ImageFolder(root=data_root)
    return dataset.classes


def get_ra_vit_checkpoint_path(checkpoint_name=RA_VIT_CHECKPOINT_NAME):
    for checkpoint_dir in RA_VIT_CHECKPOINT_DIRS:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path

    searched_paths = [str(path / checkpoint_name) for path in RA_VIT_CHECKPOINT_DIRS]
    raise FileNotFoundError(f"Could not find RA-ViT checkpoint. Searched: {searched_paths}")


def select_cropped_images(data_root=CROPPED_DATA_ROOT, species_name=None, limit=5):
    data_root = Path(data_root)

    if species_name is not None:
        image_paths = sorted((data_root / species_name).glob("*.jpg"))
    else:
        image_paths = sorted(data_root.glob("*/*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No cropped images found in {data_root}")

    return image_paths[:limit]


def load_single_image(image_path, device=None):
    if device is None:
        device = get_device()

    image = Image.open(image_path).convert("RGB")
    image_tensor = vit_b16_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(device)


def build_ra_vit_model(
    num_classes,
    checkpoint_path=None,
    fc1_dim=512,
    dropout=0.3,
    device=None,
):
    if device is None:
        device = get_device()

    if checkpoint_path is None:
        checkpoint_path = get_ra_vit_checkpoint_path()

    model = RA_ViT(
        num_classes=num_classes,
        fc1_dim=fc1_dim,
        dropout=dropout,
        freeze_backbones=True,
    )
    model_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    return model


def get_ra_vit_logits(model, image_path, device=None):
    if device is None:
        device = get_device()

    image_tensor = load_single_image(image_path, device=device)

    with torch.no_grad():
        global_logits, local_logits, total_logits = model(image_tensor)

    return global_logits, local_logits, total_logits


def get_top_predictions(logits, class_names, top_k=3):
    probabilities = torch.softmax(logits, dim=1)
    top_probabilities, top_indices = probabilities.topk(top_k, dim=1)

    predictions = []
    for probability, class_index in zip(top_probabilities[0], top_indices[0]):
        predictions.append({
            "class_index": class_index.item(),
            "class_name": class_names[class_index.item()],
            "probability": probability.item(),
        })

    return predictions


def print_top_predictions(title, logits, class_names, top_k=3):
    print(title)
    predictions = get_top_predictions(logits, class_names, top_k=top_k)
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}. {prediction['class_name']} "
            f"({prediction['probability']:.4f})"
        )


def classify_cropped_image(model, image_path, class_names, top_k=3, device=None):
    global_logits, local_logits, total_logits = get_ra_vit_logits(
        model,
        image_path,
        device=device,
    )

    print(f"Image: {image_path}")
    print_top_predictions("Global stream top predictions:", global_logits, class_names, top_k=top_k)
    print_top_predictions("Local stream top predictions:", local_logits, class_names, top_k=top_k)
    print_top_predictions("Total logits top predictions:", total_logits, class_names, top_k=top_k)

    return global_logits, local_logits, total_logits


if __name__ == "__main__":
    device = get_device()
    class_names = get_class_names()
    checkpoint_path = get_ra_vit_checkpoint_path()
    model = build_ra_vit_model(
        num_classes=len(class_names),
        checkpoint_path=checkpoint_path,
        fc1_dim=512,
        dropout=0.3,
        device=device,
    )

    image_paths = select_cropped_images(species_name="073.Blue_Jay", limit=1)
    classify_cropped_image(
        model,
        image_paths[0],
        class_names,
        top_k=3,
        device=device,
    )
