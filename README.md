# Bird ID

A bird classifier built with deep-learning models using the CUB-200-2011
dataset.

## Data preprocessing

### `data_preprocessing.py`

Extracts the CUB archive if necessary, crops images around their bounding
boxes, and saves the processed dataset and metadata under
`src/CUB_200_2011_cropped_square/`.

- Arguments: none.

### `data_cleaning.py`

Finds undersized or blurry images and optionally creates blur-analysis plots.
Deletion is only performed when `--apply` is supplied.

- `--data-root PATH`: use a different ImageFolder-style dataset.
- `--apply`: permanently delete images selected by the size or blur checks.
- `--plot-blur`: save a histogram of Laplacian blur scores.
- `--blur-output PATH`: set the blur histogram output path.
- `--blur-bins N`: set the number of histogram bins.
- `--plot-blur-samples`: save example images from different blur levels.
- `--blur-sample-output PATH`: set the blur-sample plot output path.
- `--blur-levels N`: set the number of blur-score groups.
- `--samples-per-level N`: set the images shown per blur group.
- `--remove-blurry-below SCORE`: select images below this blur score; combine
  with `--apply` to delete them.

### `data_stats.py`

Prints dataset and class-distribution statistics and saves class-count and
image-size plots under `produced_visuals/data_set_distributions/`.

- Arguments: none.

### `data_splitting.py`

Creates stratified 70/15/15 training, validation, and test dataloaders and
prints their sizes when run directly.

- Command-line arguments: none.
- Function arguments for `create_vit_b16_dataloaders()`: `data_root`,
  `batch_size`, `num_workers`, `seed`, `train_ratio`, `val_ratio`, and
  `test_ratio`.
