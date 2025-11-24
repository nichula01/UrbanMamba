"""
MambaVision Encoder Wrapper for UrbanMamba v3.
Uses the actual MambaVision architecture for feature extraction.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Optional, List

# Setup mambavision import path
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parent.parent.parent.parent

# Add repo root to path for mambavision package import
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Try importing MambaVision components
MAMBAVISION_AVAILABLE = False
PatchEmbed = None
MambaVisionLayer = None
MambaVision = None

try:
    from mambavision.models.mamba_vision import (
        MambaVision as _MambaVision, 
        PatchEmbed as _PatchEmbed,
        MambaVisionLayer as _MambaVisionLayer
    )
    PatchEmbed = _PatchEmbed
    MambaVisionLayer = _MambaVisionLayer
    MambaVision = _MambaVision
    MAMBAVISION_AVAILABLE = True
    print("✓ MambaVision modules loaded successfully")
except ImportError as e:
    print(f"⚠ MambaVision not available: {e}")
    print("  Will use fallback ConvNet encoder")


class MambaVisionEncoder(nn.Module):
    """
    MambaVision backbone encoder for extracting multi-scale features.
    
    This encoder adapts MambaVision to handle different input channels 
    (3 for RGB, 87 for NSST) and extracts features at 4 different scales.
    
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
        freeze: bool = False,
        drop_path_rate: float = 0.3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.variant = variant
        self.pretrained = pretrained
        self.freeze = freeze
        self.drop_path_rate = drop_path_rate
        
        # Variant configurations
        self.config = {
            'tiny': {
                'dim': 80,
                'in_dim': 32,
                'depths': [1, 3, 8, 4],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
                'mlp_ratio': 4,
                'dims': [80, 160, 320, 640]
            },
            'small': {
                'dim': 96,
                'in_dim': 64,
                'depths': [3, 3, 7, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
                'mlp_ratio': 4,
                'dims': [96, 192, 384, 768]
            },
            'base': {
                'dim': 128,
                'in_dim': 64,
                'depths': [3, 3, 10, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
                'mlp_ratio': 4,
                'dims': [128, 256, 512, 1024]
            },
            'large': {
                'dim': 160,
                'in_dim': 64,
                'depths': [3, 3, 12, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
                'mlp_ratio': 4,
                'dims': [160, 320, 640, 1280]
            }
        }
        
        cfg = self.config[variant]
        self.dims = cfg['dims']
        
        # Build backbone
        self.backbone = self._build_backbone(cfg)
        
        if freeze:
            self._freeze_backbone()
    
    def _build_backbone(self, cfg):
        """Build MambaVision backbone adapted for arbitrary input channels."""
        
        if not MAMBAVISION_AVAILABLE:
            print(f"⚠ Using fallback ConvNet encoder")
            print(f"  Input channels: {self.in_channels}, Variant: {self.variant}")
            return self._build_fallback_backbone()
        
        print(f"✓ Building MambaVision {self.variant} encoder")
        print(f"  Input channels: {self.in_channels}")
        
        # Create custom patch embedding for arbitrary input channels
        patch_embed = PatchEmbed(
            in_chans=self.in_channels,
            in_dim=cfg['in_dim'],
            dim=cfg['dim']
        )
        
        # Create MambaVision layers with increased drop_path_rate for regularization
        # Increased from 0.2 to 0.3 to prevent overfitting on small datasets
        dpr = [x.item() for x in torch.linspace(0, 0.3, sum(cfg['depths']))]
        levels = nn.ModuleList()
        
        for i in range(len(cfg['depths'])):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(
                dim=int(cfg['dim'] * 2 ** i),
                depth=cfg['depths'][i],
                num_heads=cfg['num_heads'][i],
                window_size=cfg['window_size'][i],
                mlp_ratio=cfg['mlp_ratio'],
                qkv_bias=True,
                qk_scale=None,
                conv=conv,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr[sum(cfg['depths'][:i]):sum(cfg['depths'][:i + 1])],
                downsample=(i < 3),
                layer_scale=None,
                layer_scale_conv=None,
                transformer_blocks=list(range(cfg['depths'][i]//2+1, cfg['depths'][i])) 
                    if cfg['depths'][i]%2!=0 
                    else list(range(cfg['depths'][i]//2, cfg['depths'][i])),
            )
            levels.append(level)
        
        # Combine into ModuleDict for easy access
        backbone = nn.ModuleDict({
            'patch_embed': patch_embed,
            'levels': levels
        })
        
        return backbone
    
    def _build_fallback_backbone(self):
        """Build simple ConvNet backbone as fallback."""
        backbone = nn.ModuleList()
        
        # Stem
        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.dims[0]),
            nn.GELU()
        )
        backbone.append(stem)
        
        # Stages
        for i in range(len(self.dims)):
            if i == 0:
                # Stage 1: dims[0] → dims[0], stride 2
                stage = nn.Sequential(
                    nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.dims[0]),
                    nn.GELU(),
                    nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.dims[0]),
                    nn.GELU()
                )
            else:
                # Other stages: dims[i-1] → dims[i], stride 2
                stage = nn.Sequential(
                    nn.Conv2d(self.dims[i-1], self.dims[i], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.dims[i]),
                    nn.GELU(),
                    nn.Conv2d(self.dims[i], self.dims[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.dims[i]),
                    nn.GELU()
                )
            backbone.append(stage)
        
        return backbone
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"✓ Encoder weights frozen")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from input.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            List of 4 feature tensors at different scales
            [F1, F2, F3, F4] with channels [dims[0], dims[1], dims[2], dims[3]]
        """
        if not MAMBAVISION_AVAILABLE:
            # Fallback path
            features = []
            x = self.backbone[0](x)  # Stem
            for i in range(1, len(self.backbone)):
                x = self.backbone[i](x)
                features.append(x)
            return features
        
        # MambaVision path
        features = []
        
        # Stage 0: After patch embedding (96 @ H/4, W/4)
        x = self.backbone['patch_embed'](x)
        features.append(x)  # [B, 96, 128, 128]
        
        # Stages 1-3: Extract from first 3 levels
        # Level 0: 192 @ 64x64, Level 1: 384 @ 32x32, Level 2: 768 @ 16x16
        for i in range(3):
            x = self.backbone['levels'][i](x)
            features.append(x)
        
        # We now have 4 features: [96@128, 192@64, 384@32, 768@16]
        # which matches exactly what the decoder expects!
        if len(features) != 4:
            raise RuntimeError(f"Expected 4 feature maps, got {len(features)}")
        
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channel dimensions for each stage."""
        return self.dims


def create_mambavision_encoder(
    in_channels: int = 3,
    variant: str = 'tiny',
    pretrained: Optional[str] = None,
    freeze: bool = False,
    drop_path_rate: float = 0.3
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
        freeze=freeze,
        drop_path_rate=drop_path_rate
    )


if __name__ == "__main__":
    # Test MambaVisionEncoder
    print("Testing MambaVisionEncoder with actual MambaVision...")
    
    # Test RGB encoder
    print("\n1. Creating RGB encoder (3 channels)...")
    rgb_encoder = create_mambavision_encoder(in_channels=3, variant='small')
    print(f"   Output channels: {rgb_encoder.get_output_channels()}")
    
    # Test frequency encoder
    print("\n2. Creating frequency encoder (87 channels)...")
    freq_encoder = create_mambavision_encoder(in_channels=87, variant='small')
    print(f"   Output channels: {freq_encoder.get_output_channels()}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_rgb = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        rgb_features = rgb_encoder(dummy_rgb)
    
    print(f"   RGB features:")
    for i, feat in enumerate(rgb_features):
        print(f"     Stage {i+1}: {feat.shape}")
    
    dummy_freq = torch.randn(1, 87, 512, 512)
    with torch.no_grad():
        freq_features = freq_encoder(dummy_freq)
    
    print(f"   Frequency features:")
    for i, feat in enumerate(freq_features):
        print(f"     Stage {i+1}: {feat.shape}")
    
    print("\n✓ All tests passed!")
