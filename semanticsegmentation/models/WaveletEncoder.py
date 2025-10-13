"""
Wavelet Encoder for UrbanMamba.

This module defines the `WaveletEncoder` class, which augments the
Vision Mamba backbone with a frequency‑domain pathway.  The input
image is decomposed using a single‑level 2‑D Haar wavelet transform
into four sub‑bands (LL, LH, HL and HH).  Each sub‑band is passed
through an identical Vision Mamba encoder (a copy of the spatial
encoder).  The outputs at each stage of the four encoders are
concatenated along the channel dimension to form multi‑scale wavelet
features.

The design follows the UrbanMamba methodology and allows the network
to capture both spatial and frequency information.  It is independent
from the spatial encoder but shares its configuration.
"""

from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn as nn

from .utils_wavelet import haar_wavelet_decompose


class WaveletEncoder(nn.Module):
    """Parallel wavelet encoder built from clones of the spatial encoder.

    Args:
        base_encoder (nn.Module): The spatial Vision Mamba encoder to be
            replicated.  It must implement a callable interface that
            accepts a tensor (B, C, H, W) and returns a list of feature
            maps at different resolutions.
    """

    def __init__(self, base_encoder: nn.Module):
        super().__init__()
        # Create four independent copies of the base encoder.  Each will
        # process one wavelet sub‑band; weights are not shared.
        self.encoders = nn.ModuleList([
            copy.deepcopy(base_encoder) for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply the wavelet transform and process each sub‑band.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: A list of fused feature maps.  The list
            length equals the number of stages in the base encoder.
        """
        ll, lh, hl, hh = haar_wavelet_decompose(x)
        # Process each sub‑band independently
        feats_list: List[List[torch.Tensor]] = []
        for encoder, subband in zip(self.encoders, (ll, lh, hl, hh)):
            feats = encoder(subband)
            if not isinstance(feats, list):
                raise RuntimeError(
                    "Expected the base encoder to return a list of feature maps"
                )
            feats_list.append(feats)
        # Concatenate the features at each stage along the channel dimension
        fused_feats: List[torch.Tensor] = []
        num_stages = len(feats_list[0])
        for stage_idx in range(num_stages):
            stage_feats = [feats_list[band_idx][stage_idx] for band_idx in range(4)]
            fused = torch.cat(stage_feats, dim=1)
            fused_feats.append(fused)
        return fused_feats
