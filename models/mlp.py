import math

import torch
import torch.nn as nn


class HeatGatedFourierFeatures(nn.Module):
    """Calibrated, heat-gated Cartesian Fourier features for model space [-1, 1]^dim."""

    def __init__(
        self,
        dim: int = 3,
        n_tiles: int = 4,
        num_features: int = 256,
        num_harmonics: float = 16.0,
    ):
        super().__init__()
        self.dim = dim
        bands_per_axis = num_features // dim
        self.num_features = bands_per_axis * dim

        min_freq = 0.5
        f0 = n_tiles / 4.0
        max_freq = f0 * num_harmonics

        log_min = math.log(min_freq * 2.0 * math.pi)
        log_max = math.log(max_freq * 2.0 * math.pi)

        radii = torch.exp(torch.linspace(log_min, log_max, bands_per_axis))

        freq_list = []
        for d in range(dim):
            vec = torch.zeros(bands_per_axis, dim)
            vec[:, d] = radii
            freq_list.append(vec)

        freqs = torch.cat(freq_list, dim=0)

        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", torch.rand(self.num_features) * 2 * math.pi)
        self.register_buffer("freq_sq_norms", (freqs**2).sum(dim=-1))
        self.scale = math.sqrt(2.0 / self.num_features)

    def forward(self, x: torch.Tensor, sigma_t: torch.Tensor):
        if sigma_t.ndim == 1:
            sigma_t = sigma_t.unsqueeze(-1)

        proj = x @ self.freqs.T + self.phases
        attenuation = torch.exp(-0.5 * self.freq_sq_norms.unsqueeze(0) * (sigma_t**2))

        return torch.cos(proj) * attenuation * self.scale


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.film_proj = nn.Linear(time_emb_dim, hidden_dim * 2)
        self.act = nn.SiLU()

        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor):
        gamma, beta = self.film_proj(t_emb).chunk(2, dim=-1)
        residual = h
        h = self.linear1(h)
        h = self.act(h * (1.0 + gamma) + beta)
        h = self.linear2(h)
        return self.act(h + residual)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        spatial_embed: HeatGatedFourierFeatures = None,
        alpha_bars: torch.Tensor = None,
        depth: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.spatial_embed = spatial_embed

        spatial_dim = 0
        if spatial_embed is not None:
            if alpha_bars is None:
                raise ValueError("alpha_bars is required when spatial_embed is set.")
            self.register_buffer("alpha_bars", alpha_bars)
            spatial_dim = spatial_embed.num_features

        # Initial projection of spatial inputs to hidden dimension
        self.in_proj = nn.Sequential(
            nn.Linear(dim + spatial_dim, hidden_dim),
            nn.SiLU(),
        )

        # Depth-configurable FiLM blocks
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden_dim, time_emb_dim) for _ in range(depth)]
        )

        # Final score/noise output projection
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_embed(t)

        if self.spatial_embed is not None:
            sigma_t = torch.sqrt(1.0 - self.alpha_bars[t])
            x_emb = self.spatial_embed(x, sigma_t)
            h = self.in_proj(torch.cat([x, x_emb], dim=-1))
        else:
            h = self.in_proj(x)

        for block in self.blocks:
            h = block(h, t_emb)

        return self.out_proj(h)


class FlowFourierFeatures(nn.Module):
    """Cartesian Fourier Features with straight-path variance gating for Flow Matching."""

    def __init__(
        self,
        dim: int = 3,
        n_tiles: int = 4,
        num_features: int = 256,
        num_harmonics: float = 16.0,
    ):
        super().__init__()
        self.dim = dim
        bands_per_axis = num_features // dim
        self.num_features = bands_per_axis * dim

        min_freq = 0.5
        f0 = n_tiles / 4.0
        max_freq = f0 * num_harmonics

        log_min = math.log(min_freq * 2.0 * math.pi)
        log_max = math.log(max_freq * 2.0 * math.pi)
        radii = torch.exp(torch.linspace(log_min, log_max, bands_per_axis))

        freq_list = []
        for d in range(dim):
            vec = torch.zeros(bands_per_axis, dim)
            vec[:, d] = radii
            freq_list.append(vec)

        freqs = torch.cat(freq_list, dim=0)

        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", torch.rand(self.num_features) * 2 * math.pi)
        self.register_buffer("freq_sq_norms", (freqs**2).sum(dim=-1))
        self.scale = math.sqrt(2.0 / self.num_features)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """In Flow Matching, effective noise variance decays as (1 - t)^2 towards data at t=1."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        proj = x @ self.freqs.T + self.phases
        # Gating decays as t:
        sigma_eff = 1.0 - t
        attenuation = torch.exp(-0.5 * self.freq_sq_norms.unsqueeze(0) * (sigma_eff**2))

        return torch.cos(proj) * attenuation * self.scale


class FlowMLP(nn.Module):
    """Velocity Field Network v_theta(x, t) for Flow Matching."""

    def __init__(
        self,
        dim: int = 3,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        num_features: int = 384,
        num_harmonics: float = 16.0,
        depth: int = 4,
        n_tiles: int = 4,
        use_rff: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        spatial_dim = 0
        self.spatial_embed = None
        if use_rff:
            self.spatial_embed = FlowFourierFeatures(
                dim=dim,
                n_tiles=n_tiles,
                num_features=num_features,
                num_harmonics=num_harmonics,
            )
            spatial_dim = self.spatial_embed.num_features

        self.in_proj = nn.Sequential(
            nn.Linear(dim + spatial_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden_dim, time_emb_dim) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # SinusoidalTimeEmbedding's frequencies are calibrated for DDPM's
        # integer t (~0-1000); CFM's continuous t in [0,1] would barely move
        # them, so rescale before embedding. FlowFourierFeatures wants the
        # raw [0,1] t (it's calibrated for that range already) -- don't touch it.
        t_emb = self.time_embed(t * 1000.0)
        if self.spatial_embed is not None:
            x_emb = self.spatial_embed(x, t)
            h = self.in_proj(torch.cat([x, x_emb], dim=-1))
        else:
            h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)
