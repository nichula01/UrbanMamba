"""
Urban Context Decoder for UrbanMamba.

The decoder progressively upsamples fused encoder features and
refines them using Urban Context Blocks.  Each block combines
features from two scales (a high‑resolution feature from the current
stage and an upsampled feature from a deeper stage), reduces the
channel dimensionality, and applies spatial and channel attention
.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class UrbanContextBlock(nn.Module):
    """Context block with spatial and channel attention.

    Args:
        in_channels (int): Number of channels in the concatenated input.
        out_channels (int): Number of channels after projection.
        reduction (int): Reduction factor for channel attention.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        squeeze_channels = max(out_channels // reduction, 1)
        self.channel_fc1 = nn.Conv2d(out_channels, squeeze_channels, kernel_size=1, bias=False)
        self.channel_fc2 = nn.Conv2d(squeeze_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        # spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * spatial_attn
        # channel attention
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        attn = self.channel_fc2(
            F.relu(self.channel_fc1(avg_pool)) + F.relu(self.channel_fc1(max_pool))
        )
        attn = torch.sigmoid(attn)
        return x * attn


class UrbanContextDecoder(nn.Module):
    """Hierarchical decoder with Urban Context Blocks.

    Args:
        in_channels_list (List[int]): Channels of each fused encoder feature,
            ordered from low to high resolution.
    """

    def __init__(self, in_channels_list: List[int]) -> None:
        super().__init__()
        if len(in_channels_list) < 1:
            raise ValueError("UrbanContextDecoder requires at least one feature map")
        self.blocks = nn.ModuleList()
        # For N features we need N–1 blocks (fusing deep to shallow).
        for idx in range(len(in_channels_list) - 1, 0, -1):
            in_ch = in_channels_list[idx] + in_channels_list[idx - 1]
            out_ch = in_channels_list[idx - 1]
            self.blocks.append(UrbanContextBlock(in_ch, out_ch))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if not feats:
            raise ValueError("Decoder input must be a non‑empty list of tensors")
        feats = list(feats)
        out_feats = [feats[-1]]
        x = feats[-1]
        # Fuse deeper to shallower
        for i, block in enumerate(self.blocks):
            shallow_idx = len(feats) - 2 - i
            shallow_feat = feats[shallow_idx]
            x = F.interpolate(x, size=shallow_feat.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, shallow_feat], dim=1))
            out_feats.append(x)
        return out_feats[::-1]
