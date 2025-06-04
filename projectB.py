from variational_network import *
from utils import *
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

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


class HeartDataset(Dataset):
    """Dataset generating noisy, undersampled k-space measurements from
    the 2dt_heart.mat file."""

    def __init__(self, mat_path="2dt_heart.mat", noise_std=NOISE_STD,
                 acceleration=ACCEL_RATE, center_fraction=0.1, sigma=10):
        super().__init__()
        data = scipy.io.loadmat(mat_path)
        self.imgs = data["imgs"].astype(np.float32)  # (H, W, T, N)
        self.noise_std = noise_std
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.sigma = sigma

    def __len__(self):
        return self.imgs.shape[3]

    def __getitem__(self, idx):
        img = self.imgs[..., idx]  # (H, W, T)
        mask = generate_undersampling_mask(
            img.shape, self.acceleration, self.center_fraction, self.sigma
        )
        s = compute_noisy_undersampled_measurements(img, mask,
                                                    sigma=self.noise_std)

        img_t = torch.from_numpy(img.transpose(2, 0, 1))  # (T, H, W)
        real = np.real(s).transpose(2, 0, 1)
        imag = np.imag(s).transpose(2, 0, 1)
        s_t = torch.from_numpy(np.stack([real, imag], axis=0))  # (2, T, H, W)
        mask_t = torch.from_numpy(mask.transpose(2, 0, 1))
        return img_t, s_t, mask_t


def create_dataloaders(batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT):
    """Utility to create train/validation dataloaders."""
    dataset = HeartDataset()
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


