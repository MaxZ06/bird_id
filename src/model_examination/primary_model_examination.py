import random
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessing.data_splitting import CROPPED_DATA_ROOT
from src.primary_model.models import RA_ViT, weighted_logit_combiner


CLASSIFIER_CHECKPOINT_PATH = (
    REPO_ROOT / "checkpoints" / "ra_vit_classifier_preprocessed_10e_comb0.5.pt"
)
COMBINER_CHECKPOINT_PATH = (
    REPO_ROOT / "checkpoints" / "weighted_combiner_lr0.001_e10"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names(data_root=CROPPED_DATA_ROOT):
    dataset = datasets.ImageFolder(root=data_root)
    return dataset.classes


def load_primary_model(
    classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    combiner_checkpoint_path=COMBINER_CHECKPOINT_PATH,
    device=None,
):
    device = device or get_device()
    class_names = get_class_names()

    classifier = RA_ViT(num_classes=len(class_names), freeze_backbones=True)
    classifier_checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
    classifier.load_state_dict(classifier_checkpoint)
    classifier.to(device)
    classifier.eval()

    combiner = weighted_logit_combiner()
    combiner_checkpoint = torch.load(combiner_checkpoint_path, map_location=device)
    combiner.load_state_dict(combiner_checkpoint)
    combiner.to(device)
    combiner.eval()

    return classifier, combiner, class_names, device


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


def top_prediction(logits, class_names):
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = probabilities.max(dim=1)
    return class_names[predicted_index.item()], confidence.item()


def predict_images(classifier, combiner, image_paths, class_names, device):
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            global_logits, local_logits = classifier(image_tensor)
            combined_logits = combiner(global_logits, local_logits)

            global_class, global_confidence = top_prediction(global_logits, class_names)
            local_class, local_confidence = top_prediction(local_logits, class_names)
            combined_class, combined_confidence = top_prediction(
                combined_logits,
                class_names,
            )

            predictions.append({
                "image": image,
                "path": image_path,
                "class_name": combined_class,
                "confidence": combined_confidence,
                "global_class_name": global_class,
                "global_confidence": global_confidence,
                "local_class_name": local_class,
                "local_confidence": local_confidence,
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


def examine_folder(classifier, combiner, class_names, device, folder):
    image_paths = find_images(folder)
    if not image_paths:
        print(f"No images found in {folder}")
        return

    selected_paths = random.sample(image_paths, k=min(3, len(image_paths)))
    predictions = predict_images(
        classifier,
        combiner,
        selected_paths,
        class_names,
        device,
    )

    for prediction in predictions:
        print(
            f"{prediction['path'].name}: "
            f"{prediction['class_name']} "
            f"({prediction['confidence']:.2%}) | "
            f"global: {prediction['global_class_name']} "
            f"({prediction['global_confidence']:.2%}) | "
            f"local: {prediction['local_class_name']} "
            f"({prediction['local_confidence']:.2%})"
        )

    display_predictions(predictions)


def main():
    classifier, combiner, class_names, device = load_primary_model()
    print(f"Loaded classifier: {CLASSIFIER_CHECKPOINT_PATH}")
    print(f"Loaded combiner: {COMBINER_CHECKPOINT_PATH}")
    print(f"Using device: {device}")

    while True:
        folder = select_image_folder()
        if folder is None:
            break

        print(f"\nSelected folder: {folder}")
        examine_folder(classifier, combiner, class_names, device, folder)

        keep_going = input("Select another folder? [y/N]: ").strip().lower()
        if keep_going != "y":
            break


if __name__ == "__main__":
    main()
