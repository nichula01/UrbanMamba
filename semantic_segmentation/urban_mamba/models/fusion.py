"""
MambaFusionBlock: Selective fusion of spatial and frequency features using Mamba mixer.
This is the core innovation of UrbanMamba v3 Twin Tower Architecture.
"""

import torch
import torch.nn as nn


class MambaFusionBlock(nn.Module):
    """
    Twin Tower Fusion Block for UrbanMamba v3.
    Fuses spatial and frequency features using Mamba state-space model for selective integration.
    
    This block takes two feature maps (spatial and frequency) with the same channel dimension
    and produces a fused output by:
    1. Concatenating the two feature maps along the channel dimension
    2. Learning an adaptive gate to blend the two branches
    3. Refining the gated base feature with a Mamba (or convolutional) mixer
    4. Residually adding the refinement back to the gated base

    If mamba-ssm is not available, falls back to convolutional fusion.
    
    Args:
        channels: Channel dimension (same for both spatial and frequency inputs)
        d_state: State dimension for Mamba mixer (default: 16)
        d_conv: Convolution kernel size for Mamba (default: 4)
        expand: Expansion factor for Mamba (default: 2)
    """
    
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.channels = channels
        self.gate_conv = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # Try to use Mamba mixer (requires mamba-ssm)
        try:
            from mamba_ssm import Mamba
            
            # Projection inside the sequence domain (C → C)
            self.proj = nn.Linear(channels, channels)
            
            # 2. Mamba Mixer: Selective state-space model
            # This is the key - it learns to selectively integrate spatial vs frequency
            self.mixer = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            
            # 3. Normalization
            self.norm = nn.LayerNorm(channels)
            
            self.use_mamba = True
            self.conv_fusion = None
            print(f"✓ MambaFusionBlock initialized with Mamba mixer (C={channels})")
            
        except ImportError:
            # Fallback: Convolutional fusion
            print(f"⚠ mamba-ssm not available, using convolutional fusion fallback")
            
            self.proj = None
            self.conv_fusion = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels)
            )
            self.use_mamba = False
    
    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and frequency features.
        
        Args:
            spatial_feat: Spatial branch features [B, C, H, W]
            freq_feat: Frequency branch features [B, C, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = spatial_feat.shape
        assert freq_feat.shape == spatial_feat.shape, \
            f"Spatial and frequency features must have same shape. Got {spatial_feat.shape} and {freq_feat.shape}"
        
        # Adaptive gating between spatial and frequency branches
        concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)
        gate = torch.sigmoid(self.gate_conv(concat_feat))
        base = spatial_feat * (1.0 - gate) + freq_feat * gate
        
        if self.use_mamba:
            # Mamba-based fusion
            # 1. Reshape gated base to sequence: [B, H*W, C]
            seq = base.flatten(2).transpose(1, 2)
            
            # 2. Project (C → C) for stability
            proj_seq = self.proj(seq)
            
            # 3. Apply Mamba mixer and normalize
            mixed_seq = self.mixer(proj_seq)
            normed_seq = self.norm(mixed_seq)
            
            # 4. Reshape back to feature map: [B, C, H, W]
            delta = normed_seq.transpose(1, 2).reshape(B, C, H, W)
            
        else:
            # Convolutional fallback refinement on the gated base
            delta = self.conv_fusion(base)
        
        if self.alpha.ndim == 0:
            fused = base + self.alpha * delta
        else:
            fused = base + self.alpha.view(1, -1, 1, 1) * delta
        
        return fused


if __name__ == "__main__":
    # Test MambaFusionBlock
    print("Testing MambaFusionBlock...")
    
    # Create fusion block
    channels = 80
    fusion = MambaFusionBlock(channels=channels)
    
    # Create dummy features
    B, H, W = 2, 64, 64
    spatial_feat = torch.randn(B, channels, H, W)
    freq_feat = torch.randn(B, channels, H, W)
    
    print(f"Input shapes: {spatial_feat.shape}, {freq_feat.shape}")
    
    # Forward pass
    with torch.no_grad():
        fused = fusion(spatial_feat, freq_feat)
    
    print(f"Output shape: {fused.shape}")
    assert fused.shape == (B, channels, H, W), "Unexpected output shape"
    
    print("✓ MambaFusionBlock test passed!")
