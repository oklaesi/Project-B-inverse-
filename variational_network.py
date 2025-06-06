import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalNetwork(nn.Module):
    def __init__(self, n_layers=10, n_filters=1, filter_size=3,
                 regulariser="vtv"):
        """Variational network for dynamic image reconstruction.

        Parameters
        ----------
        n_layers : int
            Number of gradient descent steps (layers).
        n_filters : int
            Number of learnable filters for the VTV regulariser.
        filter_size : int
            Spatial size of the learnable filters.
        regulariser : {"vtv", "tv", "tikhonov"}
            Type of regularisation to apply. Defaults to ``"vtv"``.
        """

        super(VariationalNetwork, self).__init__()
        self.n_layers = n_layers
        self.regulariser = regulariser.lower()

        if self.regulariser not in {"vtv", "tv", "tikhonov"}:
            raise ValueError(
                "regulariser must be one of 'vtv', 'tv', or 'tikhonov'")

        # Learnable step sizes and regularization weights
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(0.9)) for _ in range(n_layers)])
        
        self.filters = nn.Parameter(torch.randn(n_layers, n_filters, filter_size, filter_size) * 0.01)
        
        # activation function (lookup table)
        self.activation_grid = torch.linspace(0, 1.5, steps=100)  # shape: (G,)
        self.activation_values = nn.Parameter(torch.rand(n_layers, n_filters, 100))  # shape: (K, F, G)

    def apply_activation(self, x, k, i):
        """
        Interpolate learnable activation function φ^{k,i} over fixed grid
        Args:
            x: input tensor (H, W)
            k: layer index
            i: filter index
        Returns:
            activated: same shape as x
        """
        grid = self.activation_grid.to(x.device)  # shape (G,)
        values = self.activation_values[k, i, :].to(x.device)  # shape (G,)

        # Normalize z to grid range [0, 1.5]
        z_clamped = x.clamp(min=0, max=1.5)

        idx = torch.bucketize(z_clamped.reshape(-1), grid)
        idx = torch.clamp(idx, 1, grid.numel() - 1)
        x0 = grid[idx - 1]
        x1 = grid[idx]
        y0 = values[idx - 1]
        y1 = values[idx]
        slope = (y1 - y0) / (x1 - x0)
        activated = y0 + slope * (z_clamped.reshape(-1) - x0)
        return activated.reshape(z_clamped.shape)
    
        
    def reg_vtv(self, x, k):
        """Compute the Vectorial Total Variation regularizer.

        The method supports inputs of shape ``(T, H, W)`` or ``(B, T, H, W)``.
        In the batched case the computation is performed independently for each
        element in the batch.
        """

        batched = x.dim() == 4
        if batched:
            B, T, H, W = x.shape
            reg_term = torch.zeros_like(x)
            x_reshaped = x.reshape(B * T, 1, H, W)
        else:
            T, H, W = x.shape
            reg_term = torch.zeros_like(x)
            x_reshaped = x.unsqueeze(1)  # (T,1,H,W)

        for i in range(self.filters.shape[1]):
            # Get the (k,i)-th filter
            filt = self.filters[k, i, :, :].unsqueeze(0).unsqueeze(0)  # (1,1,f,f)

            # Convolve each frame (batch-wise convolution)
            filtered = F.conv2d(
                x_reshaped,
                filt,
                padding="same",
                groups=1,
            )
            if batched:
                filtered_frames = filtered.view(B, T, H, W)
            else:
                filtered_frames = filtered.squeeze(1)  # (T,H,W)

            # Compute joint gradient magnitude across time
            if batched:
                magnitude = (filtered_frames ** 2).sum(dim=1) / T  # (B,H,W)
            else:
                magnitude = (filtered_frames ** 2).sum(dim=0) / T  # (H,W)
            magnitude = torch.sqrt(magnitude + 1e-8)

            # Apply learnable activation function
            phi = self.apply_activation(magnitude, k, i)  # (H, W)

            # Transpose operation: convolution with flipped filter
            flipped_filt = torch.flip(filt, dims=[2, 3])

            for t in range(T):
                if batched:
                    tmp = filtered_frames[:, t] * phi  # (B,H,W)
                    tmp_conv = F.conv2d(
                        tmp.unsqueeze(1),
                        flipped_filt,
                        padding="same",
                    ).squeeze(1)  # (B,H,W)
                    reg_term[:, t] += tmp_conv
                else:
                    tmp = filtered_frames[t] * phi  # (H,W)
                    tmp_conv = F.conv2d(
                        tmp.unsqueeze(0).unsqueeze(0),
                        flipped_filt,
                        padding="same",
                    ).squeeze(0).squeeze(0)
                    reg_term[t] += tmp_conv


        return reg_term

    def reg_tv(self, x):
        """Isotropic total variation regularizer.

        This implementation works for inputs of shape ``(T, H, W)`` as well
        as batched tensors with shape ``(B, T, H, W)``.
        """

        # Finite differences along the spatial dimensions (H, W)
        grad_x = x[..., 1:] - x[..., :-1]
        grad_y = x[..., 1:, :] - x[..., :-1, :]

        # ``torch.nn.functional.pad`` with ``mode='replicate'`` does not support
        # tensors with more than three dimensions.  Since ``x`` can be either a
        # single sequence ``(T, H, W)`` or a batched tensor ``(B, T, H, W)``, we
        # replicate the border values manually to keep the original size.

        grad_x = torch.cat([grad_x, grad_x[..., -1:]], dim=-1)
        grad_y = torch.cat([grad_y, grad_y[..., -1:, :]], dim=-2)

        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_x_norm = grad_x / magnitude
        grad_y_norm = grad_y / magnitude

        div_x = grad_x_norm - torch.cat(
            [grad_x_norm[..., :1], grad_x_norm[..., :-1]], dim=-1
        )
        div_y = grad_y_norm - torch.cat(
            [grad_y_norm[..., :1, :], grad_y_norm[..., :-1, :]], dim=-2
        )

        return div_x + div_y

    def reg_tikhonov(self, x):
        """
        Tikhonov regularizer implemented via a discrete Laplacian.

        Args:
            x: input tensor of shape ``(T, H, W)``

        Returns:
            reg_term: Laplacian of ``x`` with the same shape as ``x``
        """

        kernel = torch.tensor([[0., -1., 0.],
                              [-1., 4., -1.],
                              [0., -1., 0.]], device=x.device, dtype=x.dtype)
        kernel = kernel.view(1, 1, 3, 3)
        reg_term = F.conv2d(x.unsqueeze(1), kernel, padding=1)
        return reg_term.squeeze(1)
    
    

    def compute_g(self, x, s, M, F, FH, k):
        """
        Compute g^k = α^k * F^H(M * (F x^k - s)) + Reg^k(x^k)
        
        Args:
            x: current iterate (T, H, W)
            s: measured data (H, W, T)
            M: sampling mask (H, W, T)
            F: forward operator (function)
            FH: adjoint operator (function)
            k: current layer index
            
        Returns:
            g: tensor of shape (T, H, W)
        """
        # 1. Apply forward Fourier transform to x (output: H, W, T)
        x_kspace = F(x)  # (H, W, T)

        # 2. Compute k-space residual
        residual = (M * x_kspace) - s  # (H, W, T)

        # 3. Backproject the residual using inverse Fourier transform
        data_term = FH(residual)  # (T, H, W)

        # 4. Compute regularization term depending on the chosen scheme
        if self.regulariser == "vtv":
            reg_term = self.reg_vtv(x, k)  # (T, H, W)
        elif self.regulariser == "tv":
            reg_term = self.reg_tv(x)
        else:  # self.regulariser == "tikhonov"
            reg_term = self.reg_tikhonov(x)

        # 5. Combine with step size α^k
        g = self.alpha[k] * (data_term + reg_term)  # (T, H, W)

        return g
    
    def update_momentum(self, m_prev, g, k):
        """
        Compute m^{k+1} = μ^{k+1} * m^k + g^k

        Args:
            m_prev: previous momentum (T, H, W)
            g: current gradient (T, H, W)
            k: current layer index

        Returns:
            m_next: updated momentum (T, H, W)
        """
        return self.mu[k] * m_prev + g

    def update_x(self, x, m_next):
        """
        Compute x^{k+1} = x^k - m^{k+1}

        Args:
            x: current estimate (T, H, W)
            m_next: updated momentum (T, H, W)

        Returns:
            x_next: updated image estimate (T, H, W)
        """
        return x - m_next


    def forward(self, x0, s, i2k, k2i, mask, return_intermediate=False):

        x = x0.clone()
        m = torch.zeros_like(x0)  # Initialize momentum

        xs = []
        for k in range(self.n_layers):

            g = self.compute_g(x, s, mask, i2k, k2i, k)
            m = self.update_momentum(m, g, k)
            x = self.update_x(x, m)

            if return_intermediate:
                xs.append(x.clone())

        if return_intermediate:
            return x, xs
        return x
