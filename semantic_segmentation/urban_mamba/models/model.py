"""
UrbanMamba v3: Twin Tower Architecture for Urban Semantic Segmentation

This is the complete v3 implementation with symmetric spatial and frequency encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from .transforms import NSSTDecomposition
    from .encoder import create_mambavision_encoder
    from .fusion import MambaFusionBlock
    from .decoder import UrbanContextDecoder
except ImportError:
    from transforms import NSSTDecomposition
    from encoder import create_mambavision_encoder
    from fusion import MambaFusionBlock
    from decoder import UrbanContextDecoder


class UrbanMambaV3(nn.Module):
    """
    UrbanMamba v3: Twin Tower Architecture for Urban Semantic Segmentation.
    
    Key Features:
    - Twin symmetric encoders (spatial RGB + frequency NSST)
    - Stage-wise selective fusion with MambaFusionBlock
    - Multi-scale aggregation decoder
    - ~87% FLOPs reduction vs processing subbands separately
    
    Architecture Flow:
    ```
    Input RGB [B, 3, H, W]
         |
         ├─────────────────────────────────┐
         │                                  │
         │                            NSST Decomposition
         │                                  │
         │                            [B, 87, H, W]
         │                                  │
         ↓                                  ↓
    Spatial Encoder              Frequency Encoder
    (MambaVision 3ch)            (MambaVision 87ch)
         │                                  │
    [F1, F2, F3, F4]            [F1', F2', F3', F4']
         │                                  │
         └─────────────┬────────────────────┘
                       │
              Stage-wise Fusion
              (MambaFusionBlock)
                       │
            [Fused1, Fused2, Fused3, Fused4]
                       │
           Multi-Scale Aggregation
                       │
                Segmentation Map
                  [B, K, H, W]
    ```
    
    Args:
        num_classes: Number of segmentation classes
        variant: MambaVision variant ('tiny', 'small', 'base', 'large')
        pretrained_spatial: Path to pretrained weights for spatial encoder
        pretrained_freq: Path to pretrained weights for frequency encoder (usually None)
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        variant: str = 'tiny',
        pretrained_spatial: Optional[str] = None,
        pretrained_freq: Optional[str] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.variant = variant
        
        print(f"\n{'='*70}")
        print(f"  Building UrbanMamba v3: Twin Tower Architecture")
        print(f"{'='*70}")
        print(f"  Variant: {variant}")
        print(f"  Classes: {num_classes}")
        
        # 1. NSST Feature Extractor (RGB → 87 frequency channels)
        print(f"\n[1/5] Creating NSST extractor...")
        self.nsst_extractor = NSSTDecomposition(
            scales=3,
            directions_profile=[2, 3, 4]
        )
        print(f"      ✓ NSST: 3 → 87 channels")
        
        # 2. Spatial Encoder (RGB: 3 channels)
        print(f"\n[2/5] Creating spatial encoder (RGB)...")
        self.spatial_encoder = create_mambavision_encoder(
            in_channels=3,
            variant=variant,
            pretrained=pretrained_spatial,
            freeze=False
        )
        spatial_dims = self.spatial_encoder.get_output_channels()
        print(f"      ✓ Spatial encoder: {spatial_dims}")
        
        # 3. Frequency Encoder (NSST: 87 channels)
        print(f"\n[3/5] Creating frequency encoder (NSST)...")
        self.freq_encoder = create_mambavision_encoder(
            in_channels=87,
            variant=variant,
            pretrained=pretrained_freq,
            freeze=False
        )
        freq_dims = self.freq_encoder.get_output_channels()
        print(f"      ✓ Frequency encoder: {freq_dims}")
        
        # Verify symmetry
        assert spatial_dims == freq_dims, \
            f"Spatial and frequency encoders must have same dimensions: {spatial_dims} != {freq_dims}"
        
        self.dims = spatial_dims
        
        # 4. Stage-wise Fusion Blocks
        print(f"\n[4/5] Creating fusion blocks...")
        self.fusions = nn.ModuleList([
            MambaFusionBlock(channels=dim) for dim in self.dims
        ])
        print(f"      ✓ {len(self.fusions)} fusion blocks: {self.dims}")
        
        # 5. Urban Context Decoder
        print(f"\n[5/5] Creating Urban Context Decoder...")
        # Calculate decoder dims (progressively reduce from deepest)
        decoder_dims = [
            self.dims[-1] // 2,  # 640 → 320
            self.dims[-2] // 2,  # 320 → 160
            self.dims[-3] // 2,  # 160 → 80
            self.dims[-4] // 2   # 80 → 40
        ]
        self.decoder = UrbanContextDecoder(
            encoder_dims=self.dims,
            decoder_dims=decoder_dims,
            num_classes=num_classes
        )
        print(f"      ✓ Urban Context Decoder: {self.dims} → {decoder_dims} → {num_classes} classes")
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*70}")
        print(f"  ✓ UrbanMamba v3 built successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"{'='*70}\n")
    
    def forward(self, rgb_image: torch.Tensor, nsst_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through twin tower architecture.
        
        Args:
            rgb_image: RGB input [B, 3, H, W]
            nsst_features: Pre-computed NSST features [B, 87, H, W] (optional)
                          If None, computed on-the-fly from rgb_image
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, _, H, W = rgb_image.shape
        
        # 1. Extract NSST features if not provided
        if nsst_features is None:
            nsst_features = self.nsst_extractor(rgb_image)
        
        # 2. Forward through twin towers
        # Spatial branch: [F1, F2, F3, F4]
        spatial_feats = self.spatial_encoder(rgb_image)
        
        # Frequency branch: [F1', F2', F3', F4']
        freq_feats = self.freq_encoder(nsst_features)
        
        # 3. Stage-wise fusion
        # Fused[i] = MambaFusionBlock(Spatial[i], Freq[i])
        fused_feats = []
        for spatial_feat, freq_feat, fusion_block in zip(spatial_feats, freq_feats, self.fusions):
            fused = fusion_block(spatial_feat, freq_feat)
            fused_feats.append(fused)
        
        # 4. Decode fused features
        logits = self.decoder(fused_feats)
        
        # 5. Upsample to original resolution
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        return logits
    
    def extract_nsst_features(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Extract NSST frequency features from RGB image.
        
        Args:
            rgb_image: RGB input [B, 3, H, W]
        
        Returns:
            NSST features [B, 87, H, W]
        """
        return self.nsst_extractor(rgb_image)
    
    def get_feature_dims(self) -> List[int]:
        """Get encoder feature dimensions at each stage."""
        return self.dims


def create_urban_mamba_v3(
    num_classes: int = 6,
    variant: str = 'tiny',
    pretrained_spatial: Optional[str] = None,
    pretrained_freq: Optional[str] = None
) -> UrbanMambaV3:
    """
    Factory function to create UrbanMamba v3 model.
    
    Args:
        num_classes: Number of segmentation classes
        variant: MambaVision variant ('tiny', 'small', 'base', 'large')
        pretrained_spatial: Path to pretrained weights for spatial encoder
        pretrained_freq: Path to pretrained weights for frequency encoder (usually None)
    
    Returns:
        UrbanMambaV3 model instance
    
    Example:
        >>> model = create_urban_mamba_v3(num_classes=6, variant='tiny')
        >>> rgb = torch.randn(2, 3, 512, 512)
        >>> output = model(rgb)  # [2, 6, 512, 512]
    """
    return UrbanMambaV3(
        num_classes=num_classes,
        variant=variant,
        pretrained_spatial=pretrained_spatial,
        pretrained_freq=pretrained_freq
    )


if __name__ == "__main__":
    # Test UrbanMamba v3
    print("\n" + "="*70)
    print("  Testing UrbanMamba v3 Twin Tower Architecture")
    print("="*70)
    
    # Create model
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_rgb = torch.randn(2, 3, 256, 256)
    print(f"Input shape: {dummy_rgb.shape}")
    
    with torch.no_grad():
        output = model(dummy_rgb)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 6, 256, 256), f"Unexpected output shape: {output.shape}"
    
    # Test NSST extraction
    print("\nTesting NSST extraction...")
    nsst_features = model.extract_nsst_features(dummy_rgb)
    print(f"NSST features shape: {nsst_features.shape}")
    assert nsst_features.shape == (2, 87, 256, 256), f"Unexpected NSST shape: {nsst_features.shape}"
    
    print("\n" + "="*70)
    print("  ✓ All tests passed!")
    print("="*70 + "\n")
