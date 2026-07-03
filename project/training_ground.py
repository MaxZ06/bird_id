from pathlib import Path

import torch

from models import RA_ViT
from train import get_device, train_combiner


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
RA_VIT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "ra_vit_classifier.pt"
COMBINER_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "linear_combiner.pt"


def load_ra_vit_model(checkpoint_path=RA_VIT_CHECKPOINT_PATH, device=None):
    device = device or get_device()
    model = RA_ViT(num_classes=200, freeze_backbones=True)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = get_device()
    ra_vit_model = load_ra_vit_model(device=device)
    train_combiner(
        classifier_model=ra_vit_model,
        batch_size=512,
        learning_rate=0.01,
        epochs=5,
        optimizer="adam",
        criterion="ce",
        device=device,
        checkpoint_path=COMBINER_CHECKPOINT_PATH,
    )
