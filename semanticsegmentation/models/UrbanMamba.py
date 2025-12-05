"""
UrbanMamba segmentation model.

This implementation realises the dual‑branch UrbanMamba architecture
outlined in the provided methodology.  The model
consists of two parallel Vision Mamba encoders (one operating in the
spatial domain and one in the wavelet domain), feature fusion at each
encoder stage, a hierarchical decoder with Urban Context Blocks, a
multi‑scale fusion module and a final classification head.  It
supports training from scratch or from pretrained Vision Mamba
weights.
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Mamba_backbone import Backbone_VSSM
from .WaveletEncoder import WaveletEncoder
from .FusionModule import FusionModule
from .UrbanContextDecoder import UrbanContextDecoder
from .MultiScaleFusion import MultiScaleFusion


class UrbanMamba(nn.Module):
    """Dual‑branch UrbanMamba network for semantic segmentation.

    Args:
        output_clf (int): Number of output classes.
        pretrained (Optional[str]): Path to a Vision Mamba checkpoint to
            initialise the spatial and wavelet encoders.  If ``None``,
            the encoders are randomly initialised.
        norm_layer (str): Normalisation layer for the Vision Mamba backbone.
        **kwargs: Additional arguments forwarded to ``Backbone_VSSM``.
    """

    def __init__(self, output_clf: int, pretrained: Optional[str] = None,
                 norm_layer: str = 'ln2d', use_nsst: bool = True, **kwargs) -> None:
        super().__init__()
        # Spatial encoder
        self.spatial_encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            norm_layer=norm_layer,
            **kwargs,
        )
        # Wavelet encoder
        base_encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.wavelet_encoder = WaveletEncoder(base_encoder, use_nsst=use_nsst)

        # Determine per‑stage channel dimensions
        dims: List[int] = [self.spatial_encoder.dims[i] for i in self.spatial_encoder.out_indices]
        # Fusion modules
        self.fusion_modules = nn.ModuleList()
        for c in dims:
            self.fusion_modules.append(FusionModule(
                spatial_channels=c,
                wavelet_channels=c * 4,
                out_channels=c
            ))

        # Decoder and multi‑scale fusion
        self.decoder = UrbanContextDecoder(dims)
        fusion_channels = min(dims)
        self.multi_scale_fusion = MultiScaleFusion(dims, fusion_channels)

        # Final classifier
        self.classifier = nn.Conv2d(fusion_channels, output_clf, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits for each class."""
        # Feature extraction
        spatial_feats = self.spatial_encoder(x)
        wavelet_feats = self.wavelet_encoder(x)
        # Stage‑wise fusion
        fused_feats: List[torch.Tensor] = []
        for f_sp, f_w, fuse in zip(spatial_feats, wavelet_feats, self.fusion_modules):
            fused_feats.append(fuse(f_sp, f_w))
        # Decode and fuse across scales
        decoded_feats = self.decoder(fused_feats)
        fused = self.multi_scale_fusion(decoded_feats)
        out = self.classifier(fused)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
