"""
NSST Encoder for UrbanMamba.

This module defines the `NSSTEncoder` class, which augments the
Vision Mamba backbone with a frequency‑domain pathway.  The input
image is decomposed using a non‑subsampled shearlet transform (NSST)
into four effective bands (one low‑frequency, three averaged
high‑frequency scales).  Each band is passed through an identical
Vision Mamba encoder (a copy of the spatial encoder).  The outputs at
each stage of the four encoders are concatenated along the channel
dimension to form multi‑scale frequency features.

The design allows the network to capture both spatial and frequency
information and shares configuration with the spatial encoder.  When
`use_nsst` is False, it falls back to the previous single‑level Haar
DWT for ablation.
"""

from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn as nn

from .utils_wavelet import haar_wavelet_decompose
from .freq.nsst_utils import NSSTDecomposer, nsst_four_bands_from_rgb


class NSSTEncoder(nn.Module):
    """Parallel NSST encoder built from clones of the spatial encoder.

    Args:
        base_encoder (nn.Module): The spatial Vision Mamba encoder to be
            replicated.  It must implement a callable interface that
            accepts a tensor (B, C, H, W) and returns a list of feature
            maps at different resolutions.
    """

    def __init__(self, base_encoder: nn.Module, use_nsst: bool = True):
        super().__init__()
        # Create four independent copies of the base encoder.  Each will
        # process one wavelet/NSST sub‑band; weights are not shared.
        self.encoders = nn.ModuleList([
            copy.deepcopy(base_encoder) for _ in range(4)
        ])
        self.use_nsst = use_nsst
        self.nsst_decomposer = NSSTDecomposer((4, 4, 8)) if use_nsst else None

    def _match_channels(self, band: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """Repeat or trim channels to match encoder input expectation."""
        in_ch = None
        if hasattr(encoder, "patch_embed"):
            pe = encoder.patch_embed
            if hasattr(pe, "proj") and hasattr(pe.proj, "in_channels"):
                in_ch = pe.proj.in_channels
            elif hasattr(pe, "in_chans"):
                in_ch = pe.in_chans
        if in_ch is None:
            in_ch = band.shape[1]
        if band.shape[1] == in_ch:
            return band
        if band.shape[1] == 1 and in_ch > 1:
            return band.repeat(1, in_ch, 1, 1)
        if band.shape[1] > in_ch:
            return band[:, :in_ch, ...]
        return band.repeat(1, in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply the frequency transform and process each sub‑band.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: A list of fused feature maps.  The list
            length equals the number of stages in the base encoder.
        """
        if self.use_nsst:
            bands = nsst_four_bands_from_rgb(x, self.nsst_decomposer)
        else:
            ll, lh, hl, hh = haar_wavelet_decompose(x)
            bands = [ll, lh, hl, hh]

        # Match channels and process each band independently
        feats_list: List[List[torch.Tensor]] = []
        for encoder, subband in zip(self.encoders, bands):
            subband = self._match_channels(subband, encoder)
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
