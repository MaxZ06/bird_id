# Bird ID

Bird ID is a fine-grained bird-classification project built with PyTorch and
trained on the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset. The primary
model is an attention-guided Vision Transformer (RA-ViT) that uses a global
image branch and an attention-guided local crop branch. A ResNet-50 classifier
is included as a baseline. The model currently only supports bird classification
of the 200 bird species in the CUB-200-2011 dataset.

## RA-ViT Model

The primary model has two pretrained ViT-B/16 branches:

- The global branch classifies the full image and produces an attention map.
- The attention map selects a square local crop.
- The local branch classifies the crop.
- Training optimizes both branch losses; inference sums their logits and
  returns softmax probabilities.
- The prediction from the summed logits outperformed the indiviual branches by 3% on accuracy in test sets

![RA-ViT model structure showing the global and local ViT-B/16 branches](demo_docs/model_structure.png)

*RA-ViT structure. The diagram illustrates the two prediction branches and
attention-guided crop; the implementation combines the branch logits before
applying softmax.*

The design was inspired by Fu, Zheng, and Mei's
[*Look Closer to See Better: Recurrent Attention Convolutional Neural Network
for Fine-Grained Image Recognition*](https://openaccess.thecvf.com/content_cvpr_2017/html/Fu_Look_Closer_to_CVPR_2017_paper.html)
(CVPR 2017). RA-CNN recursively focuses on discriminative regions at finer
scales. This project adapts that coarse-to-fine global/local concept to two
ViT-B/16 branches, using transformer attention to select the local crop; it is
not an implementation of the recurrent CNN architecture from the paper.

Initial training freezes both ViT backbones and trains the classifier heads.
Fine-tuning can unfreeze the final transformer blocks in both backbones.

## Project Status

The repository contains the data-preparation, training, checkpoint-loading,
and interactive prediction code used by the project. Datasets, generated crops,
training logs, plots, and model checkpoints are intentionally excluded from Git.

## Desktop GUI

The desktop interface uses PyQt6 from `requirements.txt`:

```bash
python main/main.py
```

Choose an RA-ViT checkpoint using **Choose checkpoint**, then select an image
with **Choose image** or drag a single image onto the preview area. Click
**Identify bird** to display the three highest combined predictions at the
bottom. Supported image formats include JPEG, PNG, BMP, WebP, and TIFF.

Cropping is optional. Click **Square crop** to start with the largest centered
square. Drag and resizing the cropping box is supported. **Use full image**
disables cropping. Crop dimensions and small-crop warnings appear below the
photo.

The GUI accepts full RA-ViT state dictionaries, either directly or inside a
`model_state_dict` checkpoint field, including checkpoints without an extension.
Classifier dimensions are inferred from the weights.

## Dataset

Download CUB-200-2011 from the
[official Caltech dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/)
and place the archive at the repository root:

```text
bird_id/
`-- CUB_200_2011.tgz
```

CUB-200-2011 contains 11,788 images from 200 bird categories with one bounding
box per image. The dataset page states that the images are restricted to
non-commercial research and educational use. The archive and extracted images
must not be committed to this repository.

## Data Preparation

Run the complete preparation pipeline from the repository root:

```bash
python scripts/prepare_data.py
```

The command:

1. Extracts `CUB_200_2011.tgz` when the raw dataset is absent.
2. Crops each image to a square around its annotated bounding box.
3. Writes the ImageFolder-style dataset to
   `src/CUB_200_2011_cropped_square/`.
4. Previews the removal of images smaller than the configured threshold.
5. Validates the class directories and image count.

Cleaning is non-destructive by default. Apply the size-based cleaning only
after reviewing the preview:

```bash
python scripts/prepare_data.py --apply-cleaning
```

Optionally select blurry images using Laplacian variance:

```bash
python scripts/prepare_data.py --remove-blurry-below 100
```

Add `--apply-cleaning` to that command only when the selected images should be
deleted. Existing crops are preserved unless `--overwrite` is supplied.

Validate an existing processed dataset without extracting, cropping, or
deleting files:

```bash
python scripts/prepare_data.py --validate-only
```

Use `python scripts/prepare_data.py --help` for custom archive, raw-data, and
output locations.

## Data Splits and Augmentation

`src/data_preprocessing/data_splitting.py` creates deterministic, stratified
70/15/15 training, validation, and test splits. The seed defaults to `42` in
the training entry points.

Training augmentation includes random resized crops, horizontal flips, color
jitter, rotation, normalization with the pretrained ViT-B/16 statistics, and
random erasing. Validation and test images are resized and normalized without
random augmentation.

## Train RA-ViT

Start classifier-head training with the default processed dataset:

```bash
python scripts/train_ra_vit.py
```

Adjust the hyperparameters as needed, below is a sample set of hyperparameters:

```bash
python scripts/train_ra_vit.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --seed 42
```

Resume classifier-head training from a compatible checkpoint:

```bash
python scripts/train_ra_vit.py \
  --resume \
  --checkpoint checkpoints/ra_vit_classifier.pt_e5
```

Fine-tune the final transformer blocks:

```bash
python scripts/train_ra_vit.py \
  --fine-tune \
  --checkpoint checkpoints/ra_vit_classifier.pt_e5 \
  --num-unfrozen-blocks 2 \
  --classifier-lr 0.0001 \
  --backbone-lr 0.00001
```

The training script validates its dataset, checkpoint, numeric arguments, and
requested device before training. It refuses to replace an existing epoch
checkpoint unless `--overwrite` is supplied. Run
`python scripts/train_ra_vit.py --help` for the complete interface.

## Train the ResNet-50 Baseline

Train a frozen-backbone ResNet-50 classifier:

```bash
python src/baseline_model/train_baseline.py \
  --epochs 5 \
  --batch-size 32 \
  --output-checkpoint checkpoints/simple_resnet50.pt
```

Fine-tune final residual blocks with separate learning rates:

```bash
python src/baseline_model/train_baseline.py \
  --fine-tune \
  --num-unfrozen-layers 1 \
  --classifier-lr 0.0001 \
  --backbone-lr 0.00001
```

Continue from saved weights with `--saved-checkpoint PATH`. Run the script
with `--help` for all optimizer, regularization, data, and output options.


## Outputs

Generated artifacts are written to ignored paths:

- `checkpoints/`: model state dictionaries
- `testing_logs/`: classifier training logs
- `produced_visuals/`: generated analysis plots
- `src/CUB_200_2011_cropped_square/`: processed dataset

Checkpoint files contain model weights only. Resumed training creates a new
optimizer rather than restoring optimizer state.

## Reproducibility Notes

- Keep the same preprocessing, class ordering, split seed, and package versions
  when comparing runs.
- Training uses ImageNet-pretrained backbones. The official CUB page warns that
  some CUB images may overlap with ImageNet or Flickr-derived pretraining data.
- Checkpoints must match the configured number of classes and classifier hidden
  dimension.
- Large datasets and checkpoints should be stored outside Git or published
  separately with clear version information.
