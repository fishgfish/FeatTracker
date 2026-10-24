"""Short-Range Temporal Feature Consistency (SRTFC).

This file implements the two stages of the short-range branch of FeatTracker:

* :class:`DFE` -- Diffusion-like Feature Enhancement. A lightweight
  conditional U-Net modulates a controlled noise injection in feature space
  to suppress underwater degradation (scattering, color attenuation, low
  contrast) while keeping the semantic content of the target.
* :class:`SRTFC` -- the short-range feature memory pool. It aligns the
  enhanced feature with a dynamic bank of recent features through
  style-adaptive normalization, temporal smoothing and gated fusion.

The full SRTFC module described in the paper applies ``DFE`` first and then
the memory pool. Both stages are kept as separate switches so that the
variants can be selected from the configuration file.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionStepEmbedding(nn.Module):
    """Continuous sinusoidal embedding of the diffusion step."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.norm.weight.device)
        else:
            t = t.to(self.norm.weight.device)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        return self.norm(emb)


class AdaptiveNorm(nn.Module):
    """Group normalization followed by a condition-dependent affine transform."""

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        """
        Args:
            x: feature sequence of shape (B, C, L).
            cond: condition vector of shape (B, cond_dim).
        """
        x = self.norm(x)
        scale = self.scale(cond).unsqueeze(-1)
        shift = self.shift(cond).unsqueeze(-1)
        return x * (1 + torch.tanh(scale)) + torch.tanh(shift)


class SequenceUNet(nn.Module):
    """Lightweight 1D U-Net that predicts a condition-aware noise residual."""

    def __init__(self, in_dim, channels, cond_dim):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, in_dim)

        self.down1 = nn.Sequential(
            nn.Conv1d(in_dim, channels[0], 3, padding=1),
            AdaptiveNorm(channels[0], cond_dim),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], 3, padding=1),
            AdaptiveNorm(channels[1], cond_dim),
            nn.GELU(),
        )

        self.mid = nn.Sequential(
            nn.Conv1d(channels[1], channels[1], 3, padding=1),
            AdaptiveNorm(channels[1], cond_dim),
            nn.GELU(),
        )

        self.up1 = nn.Sequential(
            nn.Conv1d(channels[1], channels[0], 3, padding=1),
            AdaptiveNorm(channels[0], cond_dim),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(channels[0], in_dim, 3, padding=1),
            AdaptiveNorm(in_dim, cond_dim),
            nn.GELU(),
        )

        self.adaptive_adjust = nn.AdaptiveAvgPool1d(1)

        self.time_weight = nn.Parameter(torch.tensor(0.1))
        self.time_proj = nn.Linear(cond_dim, in_dim)

    def forward(self, x, t_emb, cond):
        """
        Args:
            x: sequence feature of shape (B, L, C).
            t_emb: step embedding of shape (1, cond_dim).
            cond: global condition vector of shape (B, cond_dim).
        """
        original_length = x.size(1)
        x = x.permute(0, 2, 1)

        processed_cond = self.cond_proj(cond).unsqueeze(-1)
        t_projected = self.time_proj(t_emb)
        t_projected = t_projected.mean(0, keepdim=True).expand(cond.size(0), -1).unsqueeze(-1)
        processed_cond = processed_cond + self.time_weight * t_projected

        x = x + processed_cond

        d1 = self.down1[0](x)
        d1 = self.down1[1](d1, cond)
        d1 = self.down1[2](d1)

        d1_pooled = self.adaptive_adjust(d1)
        d2 = self.down2[0](d1_pooled)
        d2 = self.down2[1](d2, cond)
        d2 = self.down2[2](d2)

        m = self.mid[0](d2)
        m = self.mid[1](m, cond)
        m = self.mid[2](m)

        u1 = self.up1[0](m)
        u1 = self.up1[1](u1, cond)
        u1 = self.up1[2](u1)

        u1 = F.interpolate(u1, size=d1_pooled.size(2), mode='linear', align_corners=False)
        u1 = u1 + d1_pooled

        u2 = self.up2[0](u1)
        u2 = self.up2[1](u2, cond)
        u2 = self.up2[2](u2)

        u2 = F.interpolate(u2, size=original_length, mode='linear', align_corners=False)
        return u2.permute(0, 2, 1)


class DFE(nn.Module):
    """Diffusion-like Feature Enhancement.

    A sequence-level MLP produces enriched features and a global condition
    vector. The conditional U-Net then predicts a noise residual, which is
    used to modulate the perturbation applied to the features. The enhanced
    sequence is finally fused with the original one through a sigmoid gate.
    """

    def __init__(self, dim=256, cond_dim=128, num_steps=700, inference_step=12,
                 stochastic_inference=True, eps=0.008):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.inference_step = inference_step
        self.stochastic_inference = stochastic_inference
        self.eps = eps

        self.seq_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        self.time_embed = DiffusionStepEmbedding(cond_dim)

        self.cond_net = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.diff_net = SequenceUNet(dim, [128, 64], cond_dim=cond_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def cosine_noise_schedule(self, t):
        """Cosine schedule controlling the noise level at step ``t``."""
        t = max(min(t, self.num_steps - 1), 0)

        device = self.cond_net[0].weight.device
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        eps_tensor = torch.tensor(self.eps, dtype=torch.float32, device=device)
        num_steps_tensor = torch.tensor(self.num_steps, dtype=torch.float32, device=device)

        t_norm = (t_tensor + eps_tensor) / (num_steps_tensor + eps_tensor)
        return torch.cos((t_norm + 0.5) * torch.pi / 2)

    def forward(self, x, t=None):
        """
        Args:
            x: input sequence of shape (B, L, C).
            t: diffusion step. Defaults to ``self.inference_step``.
        """
        if t is None:
            t = self.inference_step

        seq_features = self.seq_proj(x)

        global_seq = torch.mean(seq_features, dim=1)
        cond = self.cond_net(global_seq)

        t_emb = self.time_embed(torch.tensor([t], device=x.device))
        noise_factor = self.cosine_noise_schedule(t).to(x.device)

        corrected_seq = self.diff_net(seq_features, t_emb, cond)
        corrected_seq = corrected_seq * (1 + noise_factor)
        if self.training or self.stochastic_inference:
            corrected_seq = corrected_seq + self.residual_weight * torch.randn_like(corrected_seq)

        gate_input = torch.cat([seq_features, corrected_seq], dim=-1)
        gate = self.fusion_gate(gate_input)
        return gate * corrected_seq + (1 - gate) * seq_features


class SRTFC(nn.Module):
    """Short-range feature memory pool of the SRTFC module.

    The module keeps a rotating bank of the most recent features, retrieves
    the top-k most similar entries by cosine similarity, and fuses them with
    the current feature. The fused result drives a style-adaptive
    normalization and a temporal smoothing branch, which are combined by a
    fixed interpolation weight.
    """

    VALID_STRATEGIES = ('default', 'ema', 'mean', 'random', 'fifo_recent')

    def __init__(self, dim, memory_size=5, strategy='default'):
        super().__init__()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                'Unknown memory update strategy {!r}. Choose from {}.'.format(strategy, self.VALID_STRATEGIES)
            )

        self.dim = dim
        self.memory_size = memory_size
        self.strategy = strategy

        self.register_buffer('style_memory', torch.zeros(memory_size, dim))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))

        self.style_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.style_gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )

        self.temporal_smoother = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        with torch.no_grad():
            self.temporal_smoother.weight.data.fill_(1 / 3)

        self.style_norm = nn.InstanceNorm1d(dim)

        self.ema_factor = 0.9
        self.recent_weight = 0.7

    def update_memory(self, current_style):
        """Update the memory bank with one of the supported strategies.

        ``default`` is the FIFO rotating buffer used by FeatTracker.
        """
        if self.strategy == 'default':
            idx = self.memory_count.item() % self.memory_size
            self.style_memory[idx] = current_style.detach().mean(0)

        elif self.strategy == 'ema':
            new_memory = (1 - self.ema_factor) * current_style.detach().mean(0) + \
                         self.ema_factor * self.style_memory.mean(0)
            self.style_memory[:] = new_memory.unsqueeze(0).expand_as(self.style_memory)

        elif self.strategy == 'mean':
            new_entry = current_style.detach().mean(0)
            updated_memory = 0.9 * self.style_memory.mean(0) + 0.1 * new_entry
            self.style_memory[:] = updated_memory.unsqueeze(0).expand_as(self.style_memory)

        elif self.strategy == 'random':
            idx = torch.randint(0, self.memory_size, (1,)).item()
            self.style_memory[idx] = current_style.detach().mean(0)

        elif self.strategy == 'fifo_recent':
            idx = self.memory_count.item() % self.memory_size
            self.style_memory[idx] = self.recent_weight * current_style.detach().mean(0) + \
                                     (1 - self.recent_weight) * self.style_memory[idx]

        if self.training:
            # The counter advances only during training, matching the released
            # checkpoints: at inference the current style overwrites the slot
            # selected by the final training counter.
            self.memory_count += 1

    def forward(self, original, corrected):
        """
        Args:
            original: feature before enhancement, shape (B, L, C).
            corrected: feature after enhancement, shape (B, L, C).
        """
        B, L, C = original.shape

        current_style = self.style_extractor(original.permute(0, 2, 1))
        self.update_memory(current_style)

        historical_styles = self.style_memory.clone()
        style_similarities = F.cosine_similarity(
            current_style.unsqueeze(1),
            historical_styles.unsqueeze(0),
            dim=-1,
        )

        topk_values, topk_indices = torch.topk(style_similarities, k=min(3, self.memory_size), dim=1)
        weights = F.softmax(topk_values, dim=1)

        selected_styles = historical_styles[topk_indices]
        fused_style = torch.sum(weights.unsqueeze(-1) * selected_styles, dim=1)

        normalized_features = self.style_norm(corrected.permute(0, 2, 1))
        normalized_features = normalized_features.permute(0, 2, 1)

        gate_input = torch.cat([
            normalized_features,
            fused_style.unsqueeze(1).expand(-1, L, -1),
        ], dim=-1)
        gate = self.style_gate(gate_input)

        style_transferred = gate * normalized_features + (1 - gate) * corrected

        smoothed_features = self.temporal_smoother(style_transferred.permute(0, 2, 1))
        smoothed_features = smoothed_features.permute(0, 2, 1)

        return 0.7 * style_transferred + 0.3 * smoothed_features

    def reset(self):
        """Reset the memory bank between sequences."""
        self.style_memory.zero_()
        self.memory_count.zero_()

    def set_strategy(self, strategy):
        """Switch the memory update strategy at runtime."""
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                'Unknown memory update strategy {!r}. Choose from {}.'.format(strategy, self.VALID_STRATEGIES)
            )
        self.strategy = strategy
