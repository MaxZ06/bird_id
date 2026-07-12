from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights

from train import get_device
from training_ground import load_ra_vit_model


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "checkpoints" / "ra_vit_classifier_preprocessed_10e_comb0.5.pt"
)
DEFAULT_IMAGE_PATH = (
    PROJECT_DIR
    / "CUB_200_2011_cropped_square"
    / "058.Pigeon_Guillemot"
    / "Pigeon_Guillemot_0026_40126.jpg"
)

def unnormalize_vit_image(tensor):
    weights = ViT_B_16_Weights.DEFAULT
    mean = torch.tensor(weights.transforms().mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def tensor_to_display_image(tensor):
    return unnormalize_vit_image(tensor.detach().cpu()).permute(1, 2, 0)


if __name__ == "__main__":
    device = get_device()
    model = load_ra_vit_model(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device=device,
    )

    original_image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        global_logits, local_logits, local_images = model(
            image_tensor,
            return_attention=True,
        )

    total_logit = global_logits + local_logits
    predicted_class = total_logit.argmax(dim=1).item()
    original_display = tensor_to_display_image(image_tensor[0])
    local_display = tensor_to_display_image(local_images[0])

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_display)
    plt.title("Original Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(local_display)
    plt.title("Attention Crop")
    plt.axis("off")

    plt.suptitle(f"Predicted class index: {predicted_class}")
    plt.tight_layout()
    plt.show()
