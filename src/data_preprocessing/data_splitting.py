from collections import defaultdict
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights


SRC_ROOT = Path(__file__).resolve().parents[1]
CROPPED_DATA_ROOT = SRC_ROOT / "CUB_200_2011_cropped_square"
VIT_B16_IMAGE_SIZE = 224
VIT_B16_WEIGHTS = ViT_B_16_Weights.DEFAULT
VIT_B16_WEIGHT_TRANSFORMS = VIT_B16_WEIGHTS.transforms()
VIT_B16_MEAN = VIT_B16_WEIGHT_TRANSFORMS.mean
VIT_B16_STD = VIT_B16_WEIGHT_TRANSFORMS.std


class RandomGaussianNoise:
    def __init__(self, mean=0.0, std=0.03, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image_tensor):
        if torch.rand(1).item() >= self.p:
            return image_tensor

        noise = torch.randn_like(image_tensor) * self.std + self.mean
        return (image_tensor + noise).clamp(0.0, 1.0)


vit_b16_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        VIT_B16_IMAGE_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02,
    ),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_B16_MEAN, std=VIT_B16_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
])


vit_b16_eval_transform = transforms.Compose([
    transforms.Resize((VIT_B16_IMAGE_SIZE, VIT_B16_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_B16_MEAN, std=VIT_B16_STD),
])


def stratified_train_val_test_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=1):
    if not torch.isclose(torch.tensor(train_ratio + val_ratio + test_ratio), torch.tensor(1.0)):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    indices_by_class = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        indices_by_class[label].append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in indices_by_class.items():
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        if n_train + n_val > n_total:
            n_val = n_total - n_train

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    all_indices = train_indices + val_indices + test_indices
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Split contains repeated images.")
    if len(all_indices) != len(dataset):
        raise RuntimeError("Split does not cover the whole dataset.")

    return train_indices, val_indices, test_indices


def create_vit_b16_dataloaders(
    data_root=CROPPED_DATA_ROOT,
    batch_size=32,
    num_workers=0,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
):
    split_dataset = datasets.ImageFolder(root=data_root, transform=vit_b16_eval_transform)
    train_indices, val_indices, test_indices = stratified_train_val_test_split(
        split_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = Subset(
        datasets.ImageFolder(root=data_root, transform=vit_b16_train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.ImageFolder(root=data_root, transform=vit_b16_eval_transform),
        val_indices,
    )
    test_dataset = Subset(
        datasets.ImageFolder(root=data_root, transform=vit_b16_eval_transform),
        test_indices,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, split_dataset.classes




if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = create_vit_b16_dataloaders(batch_size=32)
    print(f"Classes: {len(class_names)}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
