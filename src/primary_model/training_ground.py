import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.primary_model.models import RA_ViT
from src.primary_model.train import get_device, train_linear_combiner, train_weighted_combiner, train_classifier


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
        / "final_stage"
        / "RA_ViT_final_ver1_e20",
        device=device,
    )

    linear_results = train_linear_combiner(
        classifier_model=ra_vit_model,
        batch_size=32,
        learning_rate=0.01,
        epochs=10,
        optimizer="adam",
        criterion="ce",
        device=device,
        checkpoint_path=(
            REPO_ROOT
            / "checkpoints"
            / "final_stage"
            / "linear_combiner_ver1.pt"
        ),
    )

    weighted_results = train_weighted_combiner(
        classifier_model=ra_vit_model,
        batch_size=32,
        learning_rate=0.005,
        epochs=10,
        optimizer="adam",
        criterion="ce",
        device=device,
        checkpoint_path=(
            REPO_ROOT
            / "checkpoints"
            / "final_stage"
            / "weighted_combiner_ver1.pt"
        ),
    )


"""   
    train_classifier(
        epochs=10,
        model=ra_vit_model,
        learning_rate=0.001,
        batch_size=32,
        checkpoint_path=Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "final_stage"
        / "RA_ViT_final_ver1_e20",
    )
"""