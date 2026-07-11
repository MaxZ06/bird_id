from pathlib import Path

import torch

from models import RA_ViT
from train import get_device, train_linear_combiner, train_weighted_combiner


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

    # testing done on a pretrained classifier model (primary model)
    device = get_device()
    ra_vit_model = load_ra_vit_model(checkpoint_path="checkpoints/ra_vit_classifier_withDataAug_10e.pt", device=device)
    train_linear_combiner(classifier_model=ra_vit_model, learning_rate=0.01, epochs=10, checkpoint_path="checkpoints/lienar_combiner_lr0.01")
