from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Stochastic depth per sample (timm-style)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class FusionModule(nn.Module):
    """
    Fusion module for UrbanMamba (per stage j).

    1) Project spatial feature S_j and wavelet feature W_j to a common width C_f via 1×1 conv.
    2) Concatenate along channels: [B, 2*C_f, H, W].
    3) Reduce back to C_f with 1×1 conv and normalisation.
    4) Apply a mixer block (e.g., a VMamba block) with residual and optional DropPath.
    """

    def __init__(
        self,
        spatial_channels: int,
        wavelet_channels: int,
        out_channels: int,                  # C_f in the paper
        norm_layer: type = nn.BatchNorm2d,
        act_layer: type = nn.ReLU,
        mixer: Optional[nn.Module] = None,  # e.g. VMambaBlock(out_channels)
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        C_f = out_channels

        # Construct activation instances; use inplace=True only if supported
        if 'inplace' in act_layer.__init__.__code__.co_varnames:
            self.act = act_layer(inplace=True)
        else:
            self.act = act_layer()

        # Projections to unify channel widths of spatial and wavelet inputs
        self.proj_s = nn.Sequential(
            nn.Conv2d(spatial_channels, C_f, kernel_size=1, bias=False),
            norm_layer(C_f),
            act_layer(inplace=True) if 'inplace' in act_layer.__init__.__code__.co_varnames else act_layer(),
        )
        self.proj_w = nn.Sequential(
            nn.Conv2d(wavelet_channels, C_f, kernel_size=1, bias=False),
            norm_layer(C_f),
            act_layer(inplace=True) if 'inplace' in act_layer.__init__.__code__.co_varnames else act_layer(),
        )

        # Concatenate and reduce back to C_f
        self.reduce = nn.Sequential(
            nn.Conv2d(2 * C_f, C_f, kernel_size=1, bias=False),
            norm_layer(C_f),
        )

        # Mixer block (e.g., a VMamba block), default Identity
        self.mixer = mixer if mixer is not None else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.pre_norm = norm_layer(C_f)

    def forward(self, spatial_feat: torch.Tensor, wavelet_feat: torch.Tensor) -> torch.Tensor:
        if spatial_feat.shape[-2:] != wavelet_feat.shape[-2:]:
            raise ValueError(
                f"Spatial and wavelet features must share H×W; "
                f"got {spatial_feat.shape[-2:]} vs {wavelet_feat.shape[-2:]}"
            )

        # Project to common width
        xs = self.proj_s(spatial_feat)   # [B, C_f, H, W]
        xw = self.proj_w(wavelet_feat)   # [B, C_f, H, W]

        # Concatenate and reduce
        x = torch.cat([xs, xw], dim=1)   # [B, 2*C_f, H, W]
        x = self.reduce(x)               # [B, C_f, H, W]

        # Mixer block with residual and stochastic depth
        y = self.mixer(self.pre_norm(x))
        out = x + self.drop_path(y)
        return self.act(out)
