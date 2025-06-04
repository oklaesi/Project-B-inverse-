from variational_network import *
from utils import *

#─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
N_LAYERS   = 10
N_FILTERS  = 8
FILTER_SZ  = 3

NOISE_STD          = 0.01
ACCEL_RATE         = 4
MASK_CENTER_RADIUS = 8

BATCH_SIZE   = 4
NUM_EPOCHS   = 50
LR           = 1e-4
PRINT_EVERY  = 10
TRAIN_SPLIT  = 0.8
#─────────────────────────────────────────────────────────────────────────────

