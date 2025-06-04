import numpy as np
import torch
import torch.nn.functional as F

def k2i(img, dims=(0, )):
    dim_img = img.shape
    if dims is None:
        factor = np.prod(dim_img)
        return np.sqrt(factor) * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(img)))
    else:
        for dim in dims:
            img = np.sqrt(dim_img[dim]) * np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(img, axes=dim), axis=dim), axes=dim
            )
        return img

def i2k(img, dims=(0, )):
    dim_img = img.shape
    if dims is None:
        factor = np.prod(dim_img)
        return (1/np.sqrt(factor)) * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    else:
        for dim in dims:
            img = (1/np.sqrt(dim_img[dim])) * np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(img, axes=dim), axis=dim), axes=dim
            )
        return img

def vtv_loss(x):
    """
    Computes the Vectorial Total Variation (VTV) loss as described in Eq. (19) of the script.

    Args:
        x (torch.Tensor): A dynamic image sequence of shape (T, H, W),
                          where T is the number of frames, H is the height, and W is the width.

    Returns:
        torch.Tensor: The VTV loss value (scalar).
    """
    T, H, W = x.shape

    # Compute spatial gradients for each frame (along spatial axes)
    grad_x = x[:, :, 1:] - x[:, :, :-1]       # horizontal gradient (along width)
    grad_y = x[:, 1:, :] - x[:, :-1, :]       # vertical gradient (along height)

    # Pad to (T, H, W) for consistent size
    grad_x = F.pad(grad_x, (0, 1), mode='replicate')   # pad last column (W dimension)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')  # pad last row (H dimension)

    # Compute squared gradient magnitude across time (sum over T)
    squared_gradients = grad_x**2 + grad_y**2
    vtv_map = squared_gradients.sum(dim=0)  # sum over T, shape: (H, W)

    # Take sqrt and sum over spatial dimensions
    vtv = torch.sqrt(vtv_map + 1e-8).sum()

    return vtv

def generate_undersampling_mask(shape, acceleration, center_fraction=0.1, sigma=10):
    """
    Generates a 3D undersampling mask with incoherent sampling across time and a Laplace-shaped density.

    Args:
        shape: (H, W, T) — dimensions of k-space data
        acceleration: desired acceleration factor (e.g., 4 or 6)
        center_fraction: fraction of central k-space to fully sample
        sigma: width parameter for Laplace-shaped density (controls decay)

    Returns:
        mask: numpy array of shape (H, W, T) with 0 (unsampled) or 1 (sampled)
    """
    H, W, T = shape
    mask = np.zeros((H, W, T), dtype=np.float32)

    # Sample the center of k-space
    center_size_h = int(H * center_fraction)
    center_size_w = int(W * center_fraction)
    ch_start, ch_end = H//2 - center_size_h//2, H//2 + center_size_h//2
    cw_start, cw_end = W//2 - center_size_w//2, W//2 + center_size_w//2

    mask[ch_start:ch_end, cw_start:cw_end, :] = 1  # fully sample center for all time frames

    # Create Laplace-shaped density for outer k-space sampling
    ky = np.arange(H) - H//2
    kx = np.arange(W) - W//2
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing='ij')
    distance = np.sqrt(ky_grid**2 + kx_grid**2)

    laplace_density = np.exp(-distance / sigma)
    laplace_density[ch_start:ch_end, cw_start:cw_end] = 0  # exclude fully sampled center

    # Normalize to sum to 1
    laplace_density /= laplace_density.sum()

    num_samples = int(H * W / acceleration) - (center_size_h * center_size_w)

    for t in range(T):
        # Flatten the density and sample points according to the probability
        flat_density = laplace_density.flatten()
        sampled_indices = np.random.choice(H * W, num_samples, replace=False, p=flat_density)
        sampled_coords = np.unravel_index(sampled_indices, (H, W))
        mask[sampled_coords[0], sampled_coords[1], t] = 1

    return mask

def compute_noisy_undersampled_measurements(img, mask, sigma=0.01):
    """
    Computes the noisy undersampled k-space measurements s = M (F x + N(0, sigma I)).

    Args:
        img (ndarray): Input image in spatial domain, shape (H, W, T)
        mask (ndarray): Binary undersampling mask, shape (H, W, T)
        sigma (float): Standard deviation of Gaussian noise

    Returns:
        s (ndarray): Noisy undersampled k-space data, shape (H, W, T)
    """
    # Compute Fourier transform of the image
    kspace_full = i2k(img, dims=(0, 1))  # apply FT along spatial dimensions only (H, W)

    # Add Gaussian noise (same shape as k-space)
    noise = np.random.normal(0, sigma, kspace_full.shape) + 1j * np.random.normal(0, sigma, kspace_full.shape)
    kspace_noisy = kspace_full + noise

    # Apply sampling mask (element-wise multiplication)
    s = mask * kspace_noisy

    return s


def complex_from_tensor(t):
    """Convert a real-valued tensor to a complex-valued tensor.

    The input is expected to have the real/imaginary components stacked in
    the first dimension, i.e. ``(2, T, H, W)`` or ``(B, 2, T, H, W)`` where ``B``
    denotes an optional batch dimension.  The returned tensor has the complex
    dimension removed and the time dimension moved to the end resulting in
    shapes ``(H, W, T)`` or ``(B, H, W, T)`` respectively.
    """

    if t.ndim == 4:
        # (2, T, H, W) -> (H, W, T)
        return torch.view_as_complex(t.permute(2, 3, 1, 0).contiguous())
    elif t.ndim == 5:
        # (B, 2, T, H, W) -> (B, H, W, T)
        return torch.view_as_complex(t.permute(0, 3, 4, 2, 1).contiguous())
    else:
        raise ValueError(
            "Tensor must have shape (2, T, H, W) or (B, 2, T, H, W)")


def tensor_from_complex(c):
    """Convert complex tensor to a real-valued representation.

    Accepts a tensor of shape ``(H, W, T)`` or ``(B, H, W, T)`` and returns a
    real tensor with the complex dimension stacked in the first position,
    resulting in shapes ``(2, T, H, W)`` or ``(B, 2, T, H, W)`` respectively.
    """

    real = torch.view_as_real(c)
    if c.ndim == 3:
        # (H, W, T, 2) -> (2, T, H, W)
        return real.permute(3, 2, 0, 1)
    elif c.ndim == 4:
        # (B, H, W, T, 2) -> (B, 2, T, H, W)
        return real.permute(0, 4, 3, 1, 2)
    else:
        raise ValueError(
            "Complex tensor must have shape (H, W, T) or (B, H, W, T)")


def i2k_torch(x):
    """Fourier transform from image to k-space.

    The function operates on tensors with shape ``(..., T, H, W)`` where the
    last three dimensions correspond to time, height and width.  The Fourier
    transform is applied over the spatial dimensions resulting in an output of
    shape ``(..., H, W, T)``.
    """

    # Move the time dimension to the end so that H and W are the last two dims
    x_c = torch.movedim(x, -3, -1)
    k = torch.fft.fftn(
        torch.fft.ifftshift(x_c, dim=(-3, -2)),
        dim=(-3, -2),
        norm="ortho",
    )
    k = torch.fft.fftshift(k, dim=(-3, -2))
    return k


def k2i_torch(k):
    """Inverse Fourier transform from k-space to image.

    Accepts tensors with shape ``(..., H, W, T)`` and returns real-valued image
    tensors with shape ``(..., T, H, W)``.  The inverse transform is performed
    over the spatial dimensions.
    """

    img = torch.fft.ifftn(
        torch.fft.ifftshift(k, dim=(-3, -2)),
        dim=(-3, -2),
        norm="ortho",
    )
    img = torch.fft.fftshift(img, dim=(-3, -2))
    img = torch.movedim(img, -1, -3).real
    return img
