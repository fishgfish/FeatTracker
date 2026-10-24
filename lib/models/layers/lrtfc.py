"""Long-Range Temporal Feature Consistency (LRTFC).

The tracker propagates a compact temporal token from one frame to the next.
Naive propagation accumulates noise and gradually erases fine details,
especially in underwater videos. LRTFC splits the historical token into
frequency bands before propagation:

1. the token is zero-padded to a minimum length and reshaped to a 2D map;
2. a 2D discrete wavelet transform separates one low-frequency component
   (stable structure) and three high-frequency components (transient detail);
3. the high-frequency components are summed and gated, while the
   low-frequency component is fused with the current token;
4. a linear projection maps the fused representation back to the token space.

Besides the Haar wavelet used by FeatTracker, the module supports Fourier
masking, fixed spatial attention and multi-scale convolution as alternative
separation operators, selectable from the configuration file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRTFC(nn.Module):
    """Wavelet-based temporal token propagation."""

    VALID_SEPARATIONS = ('haar', 'fourier', 'attention', 'inception')

    def __init__(self, dim, separation='haar'):
        super().__init__()
        if separation not in self.VALID_SEPARATIONS:
            raise ValueError(
                'Unknown separation mode {!r}. Choose from {}.'.format(separation, self.VALID_SEPARATIONS)
            )

        self.dim = dim
        self.separation = separation

        self._init_haar_kernels()
        self._init_frequency_mask()
        self._init_spatial_attention()
        self._init_inception_kernels()

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 5, dim * 2),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(dim * 2, dim)

    def set_separation(self, separation):
        """Switch the token separation operator at runtime."""
        if separation not in self.VALID_SEPARATIONS:
            raise ValueError(
                'Unknown separation mode {!r}. Choose from {}.'.format(separation, self.VALID_SEPARATIONS)
            )
        self.separation = separation
        return self

    def _init_haar_kernels(self):
        """Haar wavelet analysis filters."""
        w_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        w_lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        w_hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        w_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        self.register_buffer('w_ll', w_ll.view(1, 1, 2, 2))
        self.register_buffer('w_lh', w_lh.view(1, 1, 2, 2))
        self.register_buffer('w_hl', w_hl.view(1, 1, 2, 2))
        self.register_buffer('w_hh', w_hh.view(1, 1, 2, 2))

    def _init_frequency_mask(self):
        """Placeholder for the Fourier branch; built lazily on first use."""
        self.register_buffer('freq_mask', None)

    def _create_frequency_mask(self, H, W, device):
        """Centre-weighted mask used by the Fourier separation."""
        if self.freq_mask is None or self.freq_mask.shape != (H, W // 2 + 1):
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W // 2 + 1, device=device)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(X ** 2 + Y ** 2)
            self.freq_mask = torch.sigmoid(5 * (0.5 - dist))
        return self.freq_mask

    def _init_spatial_attention(self):
        """Fixed identity attention used by the attention separation."""
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=1,
            batch_first=True,
            bias=False,
        )
        for param in self.spatial_attn.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self.spatial_attn.in_proj_weight.data.zero_()
            self.spatial_attn.in_proj_weight.data[:self.dim, :] = torch.eye(self.dim)
            self.spatial_attn.out_proj.weight.data = torch.eye(self.dim)

    def _init_inception_kernels(self):
        """Fixed multi-scale convolution kernels used by the inception separation."""
        low_kernel = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype=torch.float32) / 16

        high_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ], dtype=torch.float32)

        high_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32) / 8

        high_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=torch.float32) / 8

        self.register_buffer('low_kernel', low_kernel.repeat(self.dim, 1, 1, 1))
        self.register_buffer('high_kernel', high_kernel.repeat(self.dim, 1, 1, 1))
        self.register_buffer('high_kernel_x', high_kernel_x.repeat(self.dim, 1, 1, 1))
        self.register_buffer('high_kernel_y', high_kernel_y.repeat(self.dim, 1, 1, 1))

    def _separate_features(self, x):
        if self.separation == 'haar':
            return self._haar_separation(x)
        if self.separation == 'fourier':
            return self._fourier_separation(x)
        if self.separation == 'attention':
            return self._attention_separation(x)
        if self.separation == 'inception':
            return self._inception_separation(x)
        raise ValueError('Unknown separation mode {!r}'.format(self.separation))

    def _sync_dtype(self, dtype):
        self.w_ll = self.w_ll.to(dtype)
        self.w_lh = self.w_lh.to(dtype)
        self.w_hl = self.w_hl.to(dtype)
        self.w_hh = self.w_hh.to(dtype)

    def _haar_separation(self, x):
        """2D Haar DWT producing one low- and three high-frequency maps."""
        B, C, H, W = x.shape
        if x.dtype != self.w_ll.dtype:
            self._sync_dtype(x.dtype)

        x_ll = F.conv2d(x, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        x_lh = F.conv2d(x, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hl = F.conv2d(x, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hh = F.conv2d(x, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return x_ll, [x_lh, x_hl, x_hh]

    def _fourier_separation(self, x):
        """Spectral masking that separates low and high frequencies."""
        B, C, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_fft = torch.view_as_real(x_fft)

        freq_mask = self._create_frequency_mask(H, W, x.device)

        low_freq = x_fft * freq_mask.unsqueeze(0).unsqueeze(0)
        low_freq = torch.view_as_complex(low_freq)
        ll = torch.fft.irfft2(low_freq, s=(H, W), norm='ortho')

        high_freq = x_fft * (1 - freq_mask.unsqueeze(0).unsqueeze(0))
        high_freq = torch.view_as_complex(high_freq)
        hh = torch.fft.irfft2(high_freq, s=(H, W), norm='ortho')

        ll = F.avg_pool2d(ll, 2)
        hh = F.avg_pool2d(hh, 2)

        return ll, [hh, hh, hh]

    def _attention_separation(self, x):
        """Fixed self-attention that separates global and local responses."""
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)

        with torch.no_grad():
            attn_output, _ = self.spatial_attn(tokens, tokens, tokens)

        ll = attn_output.mean(dim=1, keepdim=True)
        ll = ll.permute(0, 2, 1).view(B, C, 1, 1)
        ll = F.interpolate(ll, size=(H // 2, W // 2), mode='nearest')

        hh = tokens - attn_output
        hh = hh.permute(0, 2, 1).view(B, C, H, W)
        hh = F.avg_pool2d(hh, 2)

        return ll, [hh, hh, hh]

    def _inception_separation(self, x):
        """Multi-scale convolution separation with energy-based reweighting."""
        B, C, H, W = x.shape

        padding = (self.low_kernel.size(2) // 2, self.low_kernel.size(3) // 2)

        ll = F.conv2d(x, self.low_kernel, padding=padding, groups=C)
        ll = F.avg_pool2d(ll, 2)

        hh = F.conv2d(x, self.high_kernel, padding=padding, groups=C).abs()
        hh = F.avg_pool2d(hh, 2)

        hh_x = F.conv2d(x, self.high_kernel_x, padding=padding, groups=C).abs()
        hh_x = F.avg_pool2d(hh_x, 2)

        hh_y = F.conv2d(x, self.high_kernel_y, padding=padding, groups=C).abs()
        hh_y = F.avg_pool2d(hh_y, 2)

        with torch.no_grad():
            feature_energy = x.abs().mean(dim=(2, 3), keepdim=True)
            ll_weight = torch.sigmoid(1.0 - feature_energy)
            hh_weight = torch.sigmoid(feature_energy - 0.5)

            ll = ll * ll_weight
            hh = hh * hh_weight
            hh_x = hh_x * hh_weight
            hh_y = hh_y * hh_weight

        return ll, [hh, hh_x, hh_y]

    def forward(self, prev_token, current_token):
        """
        Args:
            prev_token: temporal token of the previous frame, shape (B, N, C).
            current_token: temporal token of the current frame, shape (B, N, C).
        """
        B, N, C = prev_token.shape

        min_tokens = 4
        prev_padded = self._pad_tokens(prev_token, min_tokens)

        H = int(prev_padded.size(1) ** 0.5)
        x = prev_padded.view(B, H, H, C).permute(0, 3, 1, 2)

        if H >= 2:
            ll, high_components = self._separate_features(x)
            lh, hl, hh = high_components
        else:
            ll = lh = hl = hh = x

        def restore(coeff):
            return coeff.permute(0, 2, 3, 1).reshape(B, -1, C)[:, :N, :]

        ll, lh, hl, hh = map(restore, [ll, lh, hl, hh])

        gate_input = torch.cat([ll, lh, hl, hh, current_token], dim=-1)
        forget_gate, update_gate = torch.chunk(self.fusion_gate(gate_input), 2, dim=-1)

        fused_high = (lh + hl + hh) * forget_gate
        new_ll = ll * update_gate + current_token * (1 - update_gate)
        return self.fusion_proj(torch.cat([new_ll, fused_high], dim=-1))

    def _pad_tokens(self, tokens, min_len):
        if tokens.size(1) < min_len:
            return F.pad(tokens, (0, 0, 0, min_len - tokens.size(1)))
        return tokens
