"""
Enhanced MambaVision Wrapper using Official MMSegmentation Implementation
Integrates the official MambaVision segmentation code with UrbanMamba's dual-branch architecture
"""

import torch
import torch.nn as nn
from typing import List, Optional
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# tools is in semantic_segmentation/tools
# mambavision is at root level
tools_dir = os.path.join(os.path.dirname(parent_dir), 'tools')
mambavision_dir = os.path.join(os.path.dirname(os.path.dirname(parent_dir)), 'mambavision')

for path in [tools_dir, mambavision_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try importing standalone MambaVision (extracted from official code)
STANDALONE_AVAILABLE = False
try:
    from .standalone_mambavision import create_standalone_mambavision
    STANDALONE_AVAILABLE = True
    print("✓ Using standalone MambaVision (based on official segmentation code)")
except ImportError as e:
    print(f"⚠ Standalone MambaVision not available: {e}")

# Try to import the official MambaVision segmentation implementation (MMSegmentation version)
OFFICIAL_MAMBAVISION_AVAILABLE = False
try:
    from mamba_vision import MM_mamba_vision
    OFFICIAL_MAMBAVISION_AVAILABLE = True
    print("✓ Using official MM_mamba_vision (MMSegmentation)")
except ImportError:
    pass

# Also try the original backbone
BACKBONE_AVAILABLE = False
try:
    from mambavision.models.mamba_vision import MambaVision
    BACKBONE_AVAILABLE = True
    print("✓ Using MambaVision backbone from pip package")
except ImportError:
    pass

if not (STANDALONE_AVAILABLE or OFFICIAL_MAMBAVISION_AVAILABLE or BACKBONE_AVAILABLE):
    raise ImportError("No MambaVision implementation available. Please install required dependencies.")


class MambaVisionSegmentationBackbone(nn.Module):
    """
    Wrapper around official MambaVision segmentation implementation.
    This uses the actual MM_mamba_vision from the semantic_segmentation/tools/ directory.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = 'mamba_vision_T',
        pretrained: str = None,
        freeze_backbone: bool = False,
        **kwargs
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB, 87 for XLET-NSST)
            backbone_name: Variant ('mamba_vision_T', 'mamba_vision_S', 'mamba_vision_B')
            pretrained: Path to pretrained weights
            freeze_backbone: Freeze backbone parameters
        """
        super(MambaVisionSegmentationBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        
        # Configurations for each variant (from official configs)
        self.configs = {
            'mamba_vision_T': {
                'depths': [1, 3, 8, 4],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 64, 32],
                'dim': 80,
                'in_dim': 32,
                'mlp_ratio': 4,
                'drop_path_rate': 0.3,
                'out_indices': [0, 1, 2, 3],
            },
            'mamba_vision_S': {
                'depths': [1, 3, 11, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 64, 32],
                'dim': 96,
                'in_dim': 48,
                'mlp_ratio': 4,
                'drop_path_rate': 0.4,
                'out_indices': [0, 1, 2, 3],
            },
            'mamba_vision_B': {
                'depths': [2, 3, 10, 5],
                'num_heads': [4, 8, 16, 32],
                'window_size': [8, 8, 64, 32],
                'dim': 128,
                'in_dim': 64,
                'mlp_ratio': 4,
                'drop_path_rate': 0.5,
                'out_indices': [0, 1, 2, 3],
            },
        }
        
        config = self.configs.get(backbone_name, self.configs['mamba_vision_T'])
        dim = config['dim']
        
        # Output dimensions: [dim, dim*2, dim*4, dim*8]
        self.out_channels = [dim, dim*2, dim*4, dim*8]
        self.num_stages = 4
        
        if not (STANDALONE_AVAILABLE or OFFICIAL_MAMBAVISION_AVAILABLE):
            raise ImportError("MambaVision implementation not available. Please install required dependencies.")
        
        self._build_official_backbone(config, in_channels)
        
        if freeze_backbone and hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Froze {backbone_name} backbone")
    
    def _build_official_backbone(self, config, in_channels):
        """Build using official/standalone MambaVision"""
        print(f"✓ Building MambaVision segmentation backbone")
        print(f"  Input channels: {in_channels}")
        print(f"  Variant: {self.backbone_name}")
        
        # Try standalone first (most compatible)
        if STANDALONE_AVAILABLE:
            variant_map = {
                'mamba_vision_T': 'tiny',
                'mamba_vision_S': 'small',
                'mamba_vision_B': 'base',
            }
            variant = variant_map.get(self.backbone_name, 'tiny')
            
            self.backbone = create_standalone_mambavision(
                in_channels=in_channels,
                variant=variant
            )
            print(f"  Using standalone implementation (based on official code)")
        
        # Fallback to official MM_mamba_vision if available
        elif OFFICIAL_MAMBAVISION_AVAILABLE:
            self.backbone = MM_mamba_vision(
                depths=config['depths'],
                num_heads=config['num_heads'],
                window_size=config['window_size'],
                dim=config['dim'],
                in_dim=config['in_dim'],
                in_chans=in_channels,
                mlp_ratio=config['mlp_ratio'],
                drop_path_rate=config['drop_path_rate'],
                norm_layer='ln2d',
                layer_scale=None,
                out_indices=config['out_indices'],
            )
            print(f"  Using official MM_mamba_vision (MMSegmentation)")
        
        else:
            # Should not reach here if called correctly
            raise RuntimeError("No MambaVision implementation available")
        
        # Load pretrained weights if available (only for 3-channel input)
        if self.pretrained and in_channels == 3:
            print(f"  Loading pretrained weights from: {self.pretrained}")
            try:
                if hasattr(self.backbone, 'init_weights'):
                    self.backbone.init_weights(self.pretrained)
                else:
                    state_dict = torch.load(self.pretrained, map_location='cpu')
                    self.backbone.load_state_dict(state_dict, strict=False)
                print("  ✓ Pretrained weights loaded")
            except Exception as e:
                print(f"  ⚠ Could not load pretrained weights: {e}")
        else:
            if in_channels != 3:
                print(f"  Training from scratch (87-channel XLET input)")
    

    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: [B, C, H, W] where C = 3 (RGB) or 87 (XLET)
        
        Returns:
            List of 4 feature maps from each stage
        """
        # Use MambaVision backbone
        features = self.backbone(x)
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channel dimensions for each stage"""
        return self.out_channels


def create_mambavision_segmentation_encoder(
    in_channels: int = 3,
    variant: str = 'tiny',
    pretrained: Optional[str] = None,
    freeze: bool = False
) -> MambaVisionSegmentationBackbone:
    """
    Factory function to create MambaVision segmentation encoder.
    
    Args:
        in_channels: 3 for RGB, 87 for XLET-NSST
        variant: 'tiny', 'small', 'base'
        pretrained: Path to pretrained checkpoint
        freeze: Freeze backbone parameters
    
    Returns:
        MambaVisionSegmentationBackbone instance
    """
    variant_map = {
        'tiny': 'mamba_vision_T',
        'small': 'mamba_vision_S',
        'base': 'mamba_vision_B',
    }
    
    backbone_name = variant_map.get(variant.lower(), 'mamba_vision_T')
    
    return MambaVisionSegmentationBackbone(
        in_channels=in_channels,
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_backbone=freeze
    )


# For backward compatibility
MambaVisionBackbone = MambaVisionSegmentationBackbone
create_mambavision_encoder = create_mambavision_segmentation_encoder
