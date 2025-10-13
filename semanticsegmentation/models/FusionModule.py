"""
Feature fusion module for UrbanMamba.

The `FusionModule` fuses spatial and wavelet features at a given stage
of the encoder.  It concatenates the two feature maps along the
channel dimension, then applies a 1×1 convolution followed by
normalisation and activation to reduce the number of channels and
facilitate interaction between modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Fuse spatial and wavelet features using a bottleneck convolution.

    Args:
        spatial_channels (int): Number of channels in the spatial feature.
        wavelet_channels (int): Number of channels in the wavelet feature.
        out_channels (int): Number of channels to produce after fusion.
        norm_layer (Callable[..., nn.Module], optional): Normalisation
            layer (default: nn.BatchNorm2d).
        act_layer (Callable[..., nn.Module], optional): Activation
            function (default: nn.ReLU).
    """

    def __init__(
        self,
        spatial_channels: int,
        wavelet_channels: int,
        out_channels: int,
        norm_layer: type = nn.BatchNorm2d,
        act_layer: type = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            spatial_channels + wavelet_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, spatial_feat: torch.Tensor, wavelet_feat: torch.Tensor) -> torch.Tensor:
        """Fuse spatial and wavelet features.

        Args:
            spatial_feat (torch.Tensor): Spatial feature (B, C_s, H, W).
            wavelet_feat (torch.Tensor): Wavelet feature (B, C_w, H, W).

        Returns:
            torch.Tensor: Fused feature map (B, out_channels, H, W).
        """
        if spatial_feat.shape[-2:] != wavelet_feat.shape[-2:]:
            raise ValueError("Spatial and wavelet features must have the same spatial dimensions")
        x = torch.cat([spatial_feat, wavelet_feat], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
