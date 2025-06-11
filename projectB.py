from variational_network import *
from utils import *
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

#─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
N_LAYERS   = 8
N_FILTERS  = 5
FILTER_SZ  = 3
REGULARISER = "tikhonov" # Options: "vtv", "tv", "tikhonov"

# Undersampling and noise parameters
NOISE_STD          = 0.05
ACCEL_RATE         = 4
CENTER_FRACTION    = 0.1
SIGMA             = 10  

# Training parameters
BATCH_SIZE   = 4
NUM_EPOCHS   = 10
LR           = 1e-2
PRINT_EVERY  = 10
TRAIN_SPLIT  = 0.8
DS_TAU       = 0.1
USE_DEEP_SUPERVISION = True
SHOW_VAL_IMAGES = True


#─────────────────────────────────────────────────────────────────────────────


class HeartDataset(Dataset):
    """Dataset generating noisy, undersampled k-space measurements from
    the 2dt_heart.mat file."""

    def __init__(self, mat_path="2dt_heart.mat", noise_std=NOISE_STD,
                 acceleration=ACCEL_RATE, center_fraction=CENTER_FRACTION, sigma=SIGMA):
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
        real = np.real(s).astype(np.float32).transpose(2, 0, 1)
        imag = np.imag(s).astype(np.float32).transpose(2, 0, 1)
        s_t = torch.from_numpy(np.stack([real, imag], axis=0))  # (2, T, H, W)
        mask_t = torch.from_numpy(mask.transpose(2, 0, 1).astype(np.float32))
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




def train_vn(num_epochs=NUM_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
             use_deep_supervision=USE_DEEP_SUPERVISION):
    """Train a variational network on the heart dataset and plot the loss.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Training batch size.
    use_deep_supervision : bool
        If ``True`` use the deep supervision loss, otherwise use the
        standard L1 loss between the network output and the ground truth.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = create_dataloaders(batch_size=batch_size)

    vn = VariationalNetwork(
        n_layers=N_LAYERS,
        n_filters=N_FILTERS,
        filter_size=FILTER_SZ,
        regulariser=REGULARISER,
    ).to(device)
    optim = torch.optim.Adam(vn.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        vn.train()
        running = 0.0
        i = 0
        for gt, s, m in train_loader:
            i += 1
            percent = (i / len(train_loader)) * 100
            print(f"Epoch {epoch+1}/{num_epochs} - Progress: {percent:.3f}%", end='\r')
            gt = gt.to(device)
            s_complex = complex_from_tensor(s).to(device)
            m = m.permute(0, 2, 3, 1).to(device)

            x0 = k2i_torch(s_complex)
            if use_deep_supervision:
                pred, preds_all = vn(
                    x0, s_complex, i2k_torch, k2i_torch, m,
                    return_intermediate=True
                )
            else:
                pred = vn(
                    x0, s_complex, i2k_torch, k2i_torch, m,
                    return_intermediate=False
                )

            plot_loss = torch.mean(torch.abs(torch.abs(pred) - torch.abs(gt)))

            if use_deep_supervision:
                K = len(preds_all)
                loss = 0.0
                for k, x_k in enumerate(preds_all, start=1):
                    weight = torch.exp(
                        torch.tensor(
                            -DS_TAU * (K - k),
                            device=gt.device,
                            dtype=torch.float32,
                        )
                    )
                    loss = loss + weight * torch.mean(torch.abs(torch.abs(x_k) - torch.abs(gt)))
            else:
                loss = plot_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            running += plot_loss.item()

        epoch_loss = running / len(train_loader)
        losses.append(epoch_loss)
        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Training Loss')
    plt.show()
    return vn, losses


def validate_vn(model, val_loader=None, batch_size=BATCH_SIZE,
                display_examples=False, num_examples=3):
    """Validate a trained variational network on the held-out set.

    Parameters
    ----------
    model : torch.nn.Module
        The trained variational network.
    val_loader : DataLoader, optional
        Validation dataloader.  If ``None``, a loader is created using
        ``create_dataloaders`` with ``batch_size``.
    batch_size : int
        Batch size to use when constructing the dataloader.
    display_examples : bool, optional
        If ``True``, display ``num_examples`` pairs of ground truth and
        reconstructed images.
    num_examples : int, optional
        Number of example pairs to display when ``display_examples`` is
        ``True``.

    Returns
    -------
    float
        The average L1 validation loss.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if val_loader is None:
        _, val_loader = create_dataloaders(batch_size=batch_size)

    model = model.to(device)
    model.eval()
    running = 0.0
    sq_error_total = 0.0
    sq_gt_total = 0.0
    examples = []
    with torch.no_grad():
        for gt, s, m in val_loader:
            gt = gt.to(device)
            s_complex = complex_from_tensor(s).to(device)
            m = m.permute(0, 2, 3, 1).to(device)

            x0 = k2i_torch(s_complex)
            pred = model(x0, s_complex, i2k_torch, k2i_torch, m)

            loss = torch.mean(torch.abs(torch.abs(pred) - torch.abs(gt)))
            running += loss.item()

            diff = torch.abs(pred) - torch.abs(gt)
            sq_error_total += torch.sum(diff ** 2).item()
            sq_gt_total += torch.sum(torch.abs(gt) ** 2).item()

            if display_examples and len(examples) < num_examples:
                for j in range(gt.shape[0]):
                    if len(examples) >= num_examples:
                        break
                    examples.append((gt[j].cpu(), pred[j].cpu()))

    if display_examples and examples:
        n = len(examples)
        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
        if n == 1:
            axes = np.array(axes).reshape(1, -1)
        for i, (gt_ex, pred_ex) in enumerate(examples):
            gt_img = torch.abs(gt_ex[0]).numpy()
            pred_img = torch.abs(pred_ex[0]).numpy()
            axes[i, 0].imshow(gt_img, cmap="gray")
            axes[i, 0].set_title("Ground Truth")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(pred_img, cmap="gray")
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

    avg_loss = running / len(val_loader)
    nrmse = np.sqrt(sq_error_total / sq_gt_total)
    print(f"Validation nRMSE: {nrmse:.6f}")
    return avg_loss


def save_trained_model(model, directory="models", filename=None, **hyperparams):
    """Save a trained model to ``directory/filename``.

    The filename will encode given hyperparameters if ``filename`` is ``None``.

    Args:
        model (torch.nn.Module): trained variational network
        directory (str): directory to store the model file
        filename (str, optional): file name for the saved state dict.  If not
            provided, a name is constructed from ``hyperparams``.
        **hyperparams: keyword arguments describing the hyperparameters used
            during training.
    """

    def _sanitize(val):
        if isinstance(val, float):
            s = f"{val:.0e}" if val < 1e-3 or val >= 1e3 else f"{val:g}"
            s = s.replace("+", "")
        else:
            s = str(val)
        return s.replace(".", "p")

    os.makedirs(directory, exist_ok=True)

    if filename is None:
        parts = [f"{k}{_sanitize(v)}" for k, v in sorted(hyperparams.items())]
        filename = "vn_" + "_".join(parts) + ".pth"

    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    model, _ = train_vn(use_deep_supervision=USE_DEEP_SUPERVISION)

    # Evaluate the trained model on the validation set
    _, val_loader = create_dataloaders(batch_size=BATCH_SIZE)
    val_loss = validate_vn(
        model,
        val_loader,
        display_examples=SHOW_VAL_IMAGES,
        num_examples=3,
    )
    print(f"Validation loss: {val_loss:.6f}")

    """
    save_trained_model(
        model,
        n_layers=N_LAYERS,
        n_filters=N_FILTERS,
        filter_size=FILTER_SZ,
        regulariser=REGULARISER,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
    )"""
