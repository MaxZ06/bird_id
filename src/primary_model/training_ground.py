import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.primary_model.models import RA_ViT, linear_combiner
from src.primary_model.train import (
    fine_tune_model,
    get_device,
    train_classifier,
    train_linear_combiner,
    train_weighted_combiner,
)


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


def load_linear_combiner_model(
    checkpoint_path=COMBINER_CHECKPOINT_PATH,
    summed_logits=400,
    hidden_layer_1=320,
    hidden_layer_2=256,
    out_logits=200,
    dropout=0.0,
    device=None,
):
    device = device or get_device()
    model = linear_combiner(
        summed_logits=summed_logits,
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
        out_logits=out_logits,
        dropout=dropout,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":

    # load current best classifier
    device = get_device()
    ra_vit_model = load_ra_vit_model(
        checkpoint_path=REPO_ROOT
        / "checkpoints"
        / "final_stage"
        / "RA_ViT_fine_tuned_v1_e14",
        device=device,
    )

    fine_tune_results = fine_tune_model(
    model=ra_vit_model,
    batch_size=16,
    epochs=10,
    num_unfrozen_blocks=2,
    classifier_learning_rate=5e-5,
    backbone_learning_rate=5e-6,
    optimizer="adamw",
    dropout=0.5,
    weight_decay=1e-4,
    device=device,
    checkpoint_path=(
        REPO_ROOT
        / "checkpoints"
        / "final_stage"
        / "RA_ViT_fine_tuned_v1_e14+"
    ),
)



# code to train linear combiner    
"""
    loaded_linear_combiner = load_linear_combiner_model(
        checkpoint_path=(
            REPO_ROOT
            / "checkpoints"
            / "final_stage"
            / "linear_combiner_ver2.21"
        ),
        dropout=0.3,
        device=device,
    )

    linear_results = train_linear_combiner(
        classifier_model=ra_vit_model,
        batch_size=32,
        combiner=loaded_linear_combiner,
        learning_rate=0.001,
        epochs=10,
        optimizer="adamw",
        criterion="ce",
        dropout=0.3,
        weight_decay=1e-4,
        device=device,
        checkpoint_path=(
            REPO_ROOT
            / "checkpoints"
            / "final_stage"
            / "linear_combiner_ver2.21_e10+"
        ),
    )
"""


"""
    weighted_results = train_weighted_combiner(
        classifier_model=ra_vit_model,
        batch_size=32,
        learning_rate=0.005,
        epochs=2,
        optimizer="adam",
        criterion="ce",
        device=device,
        checkpoint_path=(
            REPO_ROOT
            / "checkpoints"
            / "final_stage"
            / "weighted_combiner_ver2.pt"
        ),
    )

    train_classifier(
        epochs=10,
        model=ra_vit_model,
        learning_rate=0.0005,
        batch_size=32,
        dropout=0.3,
        weight_decay=1e-4,
        checkpoint_path=Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "final_stage"
        / "RA_ViT_final_ver2",
    )
"""


