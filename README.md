# Project B Variational Network

This repository contains a PyTorch implementation of a variational network for dynamic image reconstruction. The main entry point is **`projectB.py`**, which trains the model on the `2dt_heart.mat` dataset and evaluates it on a validation split.

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset file `2dt_heart.mat` in the project directory (or provide the path when constructing `HeartDataset`).

## Running the script

Training and evaluation can be executed directly from the command line:

```bash
python projectB.py
```

During training, the script prints progress information and plots the training loss after completion. After the model finishes training, it is evaluated on the validation set and example reconstructions are displayed if `SHOW_VAL_IMAGES` is set to `True`.

## Changing hyperparameters

All default hyperparameters are defined at the top of `projectB.py`:

```python
# Network parameters
N_LAYERS = 8
N_FILTERS = 5
FILTER_SZ = 3
REGULARISER = "vtv"  # Options: "vtv", "tv", "tikhonov"

# Undersampling and noise parameters
NOISE_STD = 0.05
ACCEL_RATE = 4
CENTER_FRACTION = 0.1
SIGMA = 10

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-2
PRINT_EVERY = 10
TRAIN_SPLIT = 0.8
DS_TAU = 0.1
USE_DEEP_SUPERVISION = True
SHOW_VAL_IMAGES = True
```

To experiment with different settings you can either edit these constants or call `train_vn` manually from a Python interpreter, passing your own values for `num_epochs`, `lr`, `batch_size`, and `use_deep_supervision`:

```python
from projectB import train_vn
model, losses = train_vn(num_epochs=20, lr=1e-3, batch_size=8, use_deep_supervision=False)
```

## Output

The `train_vn` function returns the trained model and a list of training losses. The script will also compute the validation loss and print the nRMSE after training.

---
