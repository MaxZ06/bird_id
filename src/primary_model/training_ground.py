import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.primary_model.models import RA_ViT
from src.primary_model.train import get_device, train_linear_combiner, train_weighted_combiner


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
    ra_vit_model = load_ra_vit_model(
        checkpoint_path=REPO_ROOT
        / "checkpoints"
        / "ra_vit_classifier_preprocessed_10e_comb0.5.pt",
        device=device,
    )
    train_weighted_combiner(
        classifier_model=ra_vit_model,
        checkpoint_path=REPO_ROOT / "checkpoints" / "weighted_combiner_lr0.001_e10",
        epochs=10,
    )
