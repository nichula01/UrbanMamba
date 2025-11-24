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
    2. Projecting down to original channel dimension
    3. Applying Mamba mixer for selective state-based fusion
    4. Layer normalization for stability
    
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
        
        # Try to use Mamba mixer (requires mamba-ssm)
        try:
            from mamba_ssm import Mamba
            
            # 1. Projection: 2*C → C (compress concatenated features)
            self.proj = nn.Linear(channels * 2, channels)
            
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
            print(f"✓ MambaFusionBlock initialized with Mamba mixer (C={channels})")
            
        except ImportError:
            # Fallback: Convolutional fusion
            print(f"⚠ mamba-ssm not available, using convolutional fusion fallback")
            
            self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)
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
        
        if self.use_mamba:
            # Mamba-based fusion
            # 1. Concatenate: [B, 2C, H, W]
            concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)
            
            # 2. Reshape to sequence: [B, H*W, 2C]
            concat_seq = concat_feat.flatten(2).transpose(1, 2)
            
            # 3. Project down: [B, H*W, C]
            proj_seq = self.proj(concat_seq)
            
            # 4. Apply Mamba mixer: [B, H*W, C]
            mixed_seq = self.mixer(proj_seq)
            
            # 5. Normalize: [B, H*W, C]
            normed_seq = self.norm(mixed_seq)
            
            # 6. Reshape back: [B, C, H, W]
            fused = normed_seq.transpose(1, 2).reshape(B, C, H, W)
            
        else:
            # Convolutional fallback
            # 1. Concatenate: [B, 2C, H, W]
            concat_feat = torch.cat([spatial_feat, freq_feat], dim=1)
            
            # 2. Project down: [B, C, H, W]
            proj_feat = self.proj(concat_feat)
            
            # 3. Convolutional fusion: [B, C, H, W]
            fused = self.conv_fusion(proj_feat)
        
        # Residual connection with spatial features (primary branch)
        fused = fused + spatial_feat
        
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
