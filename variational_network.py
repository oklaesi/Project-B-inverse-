import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalNetwork(nn.Module):
    def __init__(self, n_layers=10, n_filters=1, filter_size=3):
        super(VariationalNetwork, self).__init__()
        self.n_layers = n_layers

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

        # Linearly interpolate
        activated = torch.interp(z_clamped, grid, values)
        return activated
    
        
    def reg_vtv(self, x, k):
        """
        Regularization term for the k-th layer.
        Args:
            x: (T, H, W)
            k: current layer index
        Returns:
            reg_term: (T, H, W)
        """
        T, H, W = x.shape
        reg_term = torch.zeros_like(x)

        for i in range(self.filters.shape[1]):
            # Get the (k,i)-th filter
            filt = self.filters[k, i, :, :].unsqueeze(0).unsqueeze(0)  # (1,1,f,f)

            # Convolve each frame (batch-wise convolution)
            filtered_frames = F.conv2d(
                x.unsqueeze(1),  # (T,1,H,W)
                filt, padding='same', groups=1
            ).squeeze(1)  # (T, H, W)

            # Compute joint gradient magnitude across time
            magnitude = (filtered_frames**2).sum(dim=0) / T  # (H, W)
            magnitude = torch.sqrt(magnitude + 1e-8)  # add small constant for stability

            # Apply learnable activation function
            phi = self.apply_activation(magnitude, k, i)  # (H, W)

            # Transpose operation: convolution with flipped filter
            flipped_filt = torch.flip(filt, dims=[2, 3])
            
            for t in range(T):
                tmp = filtered_frames[t] * phi  # (H, W)
                tmp_conv = F.conv2d(
                    tmp.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
                    flipped_filt, padding='same'
                ).squeeze(0).squeeze(0)  # (H, W)

                reg_term[t] += tmp_conv


        return reg_term
    
    

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

        # 4. Compute regularization term
        reg_term = self.reg_vtv(x, k)  # (T, H, W)

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


    def forward(self, x0, s, i2k, k2i, mask):
        
        x = x0.clone()
        m = torch.zeros_like(x0)  # Initialize momentum

        for k in range(self.n_layers):
            
            g = self.compute_g(x0, s, mask, i2k, k2i, k)
            m = self.update_momentum(m, g, k)
            x = self.update_x(x, m)

        return x
