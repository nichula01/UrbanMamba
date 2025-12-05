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
from .NSSTEncoder import NSSTEncoder
from .FusionModule import FusionModule
from .UrbanContextDecoder import UrbanContextDecoder
from .MultiScaleFusion import MultiScaleFusion
from .freq.nsst_cnn_encoder import NSSTFreqEncoderCNN


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
                 norm_layer: str = 'ln2d', use_nsst: bool = True,
                 freq_encoder_type: str = "cnn", cfg=None, **kwargs) -> None:
        super().__init__()
        self.freq_encoder_type = freq_encoder_type
        self.cfg = cfg
        # Spatial encoder
        self.spatial_encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            norm_layer=norm_layer,
            **kwargs,
        )
        # Frequency encoder
        if freq_encoder_type.lower() == "vmamba":
            base_encoder = Backbone_VSSM(
                out_indices=(0, 1, 2, 3),
                pretrained=pretrained,
                norm_layer=norm_layer,
                **kwargs,
            )
            self.freq_encoder = NSSTEncoder(base_encoder, use_nsst=use_nsst)
        elif freq_encoder_type.lower() == "cnn":
            # Channels per stage should align with spatial encoder dims * 4 (previous fusion expectation)
            dummy_dims: List[int] = [self.spatial_encoder.dims[i] for i in self.spatial_encoder.out_indices]
            freq_out_channels = tuple([c * 4 for c in dummy_dims])
            self.freq_encoder = NSSTFreqEncoderCNN(
                directions_per_scale=(4, 4, 8),
                in_channels=17,
                out_channels_list=freq_out_channels,
            )
        else:
            raise ValueError(f"Unknown frequency encoder type: {freq_encoder_type}")

        # Determine per‑stage channel dimensions
        dims: List[int] = [self.spatial_encoder.dims[i] for i in self.spatial_encoder.out_indices]
        if freq_encoder_type.lower() == "cnn":
            freq_channels = [c * 4 for c in dims]
        else:
            freq_channels = [c * 4 for c in dims]
        # Fusion modules
        self.fusion_modules = nn.ModuleList()
        use_branch_gating = False
        if cfg is not None and hasattr(cfg, "FUSION"):
            use_branch_gating = getattr(cfg.FUSION, "USE_BRANCH_GATING", False)
            gate_reduction = getattr(cfg.FUSION, "GATE_REDUCTION", 4)
        else:
            gate_reduction = 4
        for c, fc in zip(dims, freq_channels):
            self.fusion_modules.append(FusionModule(
                spatial_channels=c,
                wavelet_channels=fc,
                out_channels=c,
                use_branch_gating=use_branch_gating,
                gate_reduction=gate_reduction,
            ))

        # Decoder and multi‑scale fusion
        self.decoder = UrbanContextDecoder(dims)
        fusion_channels = min(dims)
        use_scale_weights = False
        if cfg is not None and hasattr(cfg, "FUSION"):
            use_scale_weights = getattr(cfg.FUSION, "USE_LEARNABLE_SCALE_WEIGHTS", False)
        self.multi_scale_fusion = MultiScaleFusion(
            dims,
            fusion_channels,
            use_learnable_scale_weights=use_scale_weights,
        )

        # Final classifier
        self.classifier = nn.Conv2d(fusion_channels, output_clf, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits for each class."""
        # Feature extraction
        spatial_feats = self.spatial_encoder(x)
        wavelet_feats = self.freq_encoder(x)
        # Stage‑wise fusion
        fused_feats: List[torch.Tensor] = []
        for f_sp, f_w, fuse in zip(spatial_feats, wavelet_feats, self.fusion_modules):
            fused_feats.append(fuse(f_sp, f_w))
        # Decode and fuse across scales
        decoded_feats = self.decoder(fused_feats)
        fused = self.multi_scale_fusion(decoded_feats)
        out = self.classifier(fused)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
