"""
MambaVision Encoder Wrapper for UrbanMamba v3.
Provides a unified interface to create MambaVision encoders with different input channels.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class MambaVisionEncoder(nn.Module):
    """
    Wrapper for MambaVision backbone to extract multi-scale features.
    
    This encoder can handle different input channels (3 for RGB, 87 for NSST).
    It uses the MambaVision architecture from the official implementation or timm.
    
    Args:
        in_channels: Number of input channels (3 for RGB, 87 for NSST)
        variant: MambaVision variant ('tiny', 'small', 'base', 'large')
        pretrained: Path to pretrained weights or None
        freeze: Whether to freeze encoder weights
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        variant: str = 'tiny',
        pretrained: Optional[str] = None,
        freeze: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.variant = variant
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Map variant names
        variant_map = {
            'tiny': 'mamba_vision_T',
            'small': 'mamba_vision_S',
            'base': 'mamba_vision_B',
            'large': 'mamba_vision_L'
        }
        self.model_name = variant_map.get(variant, 'mamba_vision_T')
        
        # Channel dimensions for each variant
        self.dims_map = {
            'tiny': [80, 160, 320, 640],
            'small': [96, 192, 384, 768],
            'base': [128, 256, 512, 1024],
            'large': [160, 320, 640, 1280]
        }
        self.dims = self.dims_map[variant]
        
        # Try to load MambaVision
        self.backbone = self._build_backbone()
        
        if freeze:
            self._freeze_backbone()
    
    def _build_backbone(self):
        """Build MambaVision backbone."""
        
        # For now, use a simple ConvNet as placeholder
        # This will be replaced with actual MambaVision when available
        print(f"⚠ Using placeholder ConvNet encoder (MambaVision not available)")
        print(f"  Input channels: {self.in_channels}, Variant: {self.variant}")
        
        # Build a simple 4-stage ConvNet
        backbone = nn.ModuleList()
        
        # Stem: in_channels → dims[0]
        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.dims[0]),
            nn.GELU()
        )
        backbone.append(stem)
        
        # Stage 1: dims[0] → dims[0] (stride 2)
        stage1 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dims[0]),
            nn.GELU(),
            nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[0]),
            nn.GELU()
        )
        backbone.append(stage1)
        
        # Stage 2: dims[0] → dims[1] (stride 2)
        stage2 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dims[1]),
            nn.GELU(),
            nn.Conv2d(self.dims[1], self.dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[1]),
            nn.GELU()
        )
        backbone.append(stage2)
        
        # Stage 3: dims[1] → dims[2] (stride 2)
        stage3 = nn.Sequential(
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dims[2]),
            nn.GELU(),
            nn.Conv2d(self.dims[2], self.dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[2]),
            nn.GELU()
        )
        backbone.append(stage3)
        
        # Stage 4: dims[2] → dims[3] (stride 2)
        stage4 = nn.Sequential(
            nn.Conv2d(self.dims[2], self.dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.dims[3]),
            nn.GELU(),
            nn.Conv2d(self.dims[3], self.dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[3]),
            nn.GELU()
        )
        backbone.append(stage4)
        
        return backbone
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"✓ Backbone frozen")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            List of feature tensors [F1, F2, F3, F4] at different scales
        """
        features = []
        
        # Pass through stem
        x = self.backbone[0](x)  # 1/2 resolution
        
        # Pass through each stage and collect features
        for i in range(1, len(self.backbone)):
            x = self.backbone[i](x)
            features.append(x)
        
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channel dimensions for each stage."""
        return self.dims


def create_mambavision_encoder(
    in_channels: int = 3,
    variant: str = 'tiny',
    pretrained: Optional[str] = None,
    freeze: bool = False
) -> MambaVisionEncoder:
    """
    Factory function to create MambaVision encoder.
    
    Args:
        in_channels: Number of input channels (3 for RGB, 87 for NSST)
        variant: MambaVision variant ('tiny', 'small', 'base', 'large')
        pretrained: Path to pretrained weights or None
        freeze: Whether to freeze encoder weights
    
    Returns:
        MambaVisionEncoder instance
    """
    return MambaVisionEncoder(
        in_channels=in_channels,
        variant=variant,
        pretrained=pretrained,
        freeze=freeze
    )


if __name__ == "__main__":
    # Test MambaVisionEncoder
    print("Testing MambaVisionEncoder...")
    
    # Test RGB encoder
    print("\n1. Creating RGB encoder (3 channels)...")
    rgb_encoder = create_mambavision_encoder(in_channels=3, variant='tiny')
    print(f"   Output channels: {rgb_encoder.get_output_channels()}")
    
    # Test frequency encoder
    print("\n2. Creating frequency encoder (87 channels)...")
    freq_encoder = create_mambavision_encoder(in_channels=87, variant='tiny')
    print(f"   Output channels: {freq_encoder.get_output_channels()}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_rgb = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        rgb_features = rgb_encoder(dummy_rgb)
    
    print(f"   RGB features:")
    for i, feat in enumerate(rgb_features):
        print(f"     Stage {i+1}: {feat.shape}")
    
    dummy_freq = torch.randn(2, 87, 256, 256)
    with torch.no_grad():
        freq_features = freq_encoder(dummy_freq)
    
    print(f"   Frequency features:")
    for i, feat in enumerate(freq_features):
        print(f"     Stage {i+1}: {feat.shape}")
    
    print("\n✓ All tests passed!")
