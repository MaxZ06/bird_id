import random
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets
from torchvision.models import ResNet50_Weights

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model.baseline_model_resnet import SimpleResNet50, get_device
from src.data_preprocessing.data_splitting import CROPPED_DATA_ROOT


CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "simple_resnet50_e10.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_class_names(data_root=CROPPED_DATA_ROOT):
    dataset = datasets.ImageFolder(root=data_root)
    return dataset.classes


def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or get_device()
    class_names = get_class_names()

    model = SimpleResNet50(num_classes=len(class_names), weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, class_names, device


def select_image_folder():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select an image folder")
    root.destroy()
    return Path(folder) if folder else None


def find_images(folder):
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def predict_images(model, image_paths, class_names, device):
    transform = ResNet50_Weights.DEFAULT.transforms()
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            logits = model(image_tensor)
            if isinstance(logits, tuple):
                logits = logits[-1]

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)
            predictions.append({
                "image": image,
                "path": image_path,
                "class_name": class_names[predicted_index.item()],
                "confidence": confidence.item(),
            })

    return predictions


def display_predictions(predictions):
    figure, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for axis, prediction in zip(axes, predictions):
        axis.imshow(prediction["image"])
        axis.axis("off")
        axis.set_title(
            f"{prediction['class_name']}\n"
            f"{prediction['confidence']:.2%}\n"
            f"{prediction['path'].name}",
            fontsize=10,
        )

    figure.tight_layout()
    plt.show()


def examine_folder(model, class_names, device, folder):
    image_paths = find_images(folder)
    if not image_paths:
        print(f"No images found in {folder}")
        return

    selected_paths = random.sample(image_paths, k=min(3, len(image_paths)))
    predictions = predict_images(model, selected_paths, class_names, device)

    for prediction in predictions:
        print(
            f"{prediction['path'].name}: "
            f"{prediction['class_name']} "
            f"({prediction['confidence']:.2%})"
        )

    display_predictions(predictions)


def main():
    model, class_names, device = load_model()
    print(f"Loaded {CHECKPOINT_PATH}")
    print(f"Using device: {device}")

    while True:
        folder = select_image_folder()
        if folder is None:
            break

        print(f"\nSelected folder: {folder}")
        examine_folder(model, class_names, device, folder)

        keep_going = input("Select another folder? [y/N]: ").strip().lower()
        if keep_going != "y":
            break


if __name__ == "__main__":
    main()
