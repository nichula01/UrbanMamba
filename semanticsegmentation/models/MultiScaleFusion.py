"""
Multi‑scale feature fusion for UrbanMamba.

This module defines the `MultiScaleFusion` class, which aggregates
decoder outputs across multiple scales.  Each input feature map is
projected to a common number of channels using a 1×1 convolution.  The
projected features are then fused using a simple top‑down approach,
and finally averaged over all scales.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Fuse multi‑scale decoder outputs into a single feature map.

    Args:
        in_channels_list (List[int]): Channels of each decoder feature,
            ordered from low to high resolution.
        fusion_channels (int): Channel dimension for the projected features.
    """

    def __init__(self, in_channels_list: List[int], fusion_channels: int) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("MultiScaleFusion requires at least one input feature")
        self.projects = nn.ModuleList([
            nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if not feats:
            raise ValueError("MultiScaleFusion expects a non‑empty list of tensors")
        projected = [proj(f) for proj, f in zip(self.projects, feats)]
        n = len(projected)
        fused = [None] * n
        fused[-1] = projected[-1]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(fused[i + 1], size=projected[i].shape[-2:], mode='bilinear', align_corners=False)
            fused[i] = projected[i] + up
        # Average all fused maps upsampled to the highest resolution
        output_size = projected[0].shape[-2:]
        out = 0
        for i in range(n):
            out += F.interpolate(fused[i], size=output_size, mode='bilinear', align_corners=False)
        return out / n
