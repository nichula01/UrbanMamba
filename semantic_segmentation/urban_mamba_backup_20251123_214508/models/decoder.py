"""
Urban Context Decoder (UCD) with Urban Context Blocks (UCB)
Implements progressive upsampling with dual attention mechanisms for
high-resolution boundary preservation in urban semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module with 7x7 convolution.
    Large receptive field ensures extended neighborhood information for
    precise reconstruction of thin linear features (roads, boundaries).
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention.
        
        Args:
            kernel_size: Convolution kernel size (default: 7 for high-res recovery)
        """
        super(SpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            2,  # AvgPool + MaxPool concatenated
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention mask.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Spatial attention map [B, 1, H, W]
        """
        # Channel-wise pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate
        pool_cat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        
        # Compute attention
        attention = self.conv(pool_cat)  # [B, 1, H, W]
        attention = self.sigmoid(attention)
        
        return attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module using both GAP and GMP.
    Focuses on WHAT semantic features are most relevant for reconstruction.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize Channel Attention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for MLP
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention mask.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Channel attention map [B, C, 1, 1]
        """
        # Global pooling
        avg_pool = self.avg_pool(x)  # [B, C, 1, 1]
        max_pool = self.max_pool(x)  # [B, C, 1, 1]
        
        # MLP for both pools
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        
        # Combine and activate
        attention = self.sigmoid(avg_out + max_out)
        
        return attention


class UrbanContextBlock(nn.Module):
    """
    Urban Context Block (UCB) with dual attention.
    Refines features during upsampling to preserve thin structures
    and ensure semantic consistency.
    
    F_enhanced = F ⊙ A_spatial ⊙ A_channel
    """
    
    def __init__(self, channels: int, spatial_kernel: int = 7, reduction: int = 16):
        """
        Initialize Urban Context Block.
        
        Args:
            channels: Number of feature channels
            spatial_kernel: Spatial attention kernel size (7x7 for thin features)
            reduction: Channel attention reduction ratio
        """
        super(UrbanContextBlock, self).__init__()
        
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)
        self.channel_attention = ChannelAttention(channels, reduction)
        
        # Additional refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dual attention refinement.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Enhanced features [B, C, H, W]
        """
        # Compute attention masks
        spatial_att = self.spatial_attention(x)  # [B, 1, H, W]
        channel_att = self.channel_attention(x)  # [B, C, 1, 1]
        
        # Apply attention via element-wise multiplication
        x_enhanced = x * spatial_att * channel_att
        
        # Additional refinement
        x_enhanced = self.refine(x_enhanced)
        
        # Residual connection
        x_enhanced = x_enhanced + x
        
        return x_enhanced


class UrbanContextDecoder(nn.Module):
    """
    Urban Context Decoder (UCD) with progressive upsampling.
    Recovers spatial resolution while integrating skip connections
    and applying UCB refinement at each level.
    
    F_up^(j) = UCB(Upsample(F^(j+1)) ⊕ F^(j))
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [96, 192, 384, 768],
        decoder_dims: List[int] = [384, 192, 96, 48],
        num_classes: int = 6
    ):
        """
        Initialize Urban Context Decoder.
        
        Args:
            encoder_dims: Channel dimensions from encoder stages (for skip connections)
            decoder_dims: Channel dimensions for decoder stages
            num_classes: Number of output classes (6 for urban segmentation)
        """
        super(UrbanContextDecoder, self).__init__()
        
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_stages = len(decoder_dims)
        
        # Build decoder stages
        self.decoder_stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            # Input: upsampled from previous + skip connection
            if i == 0:
                in_channels = encoder_dims[-1]  # Deepest encoder features
            else:
                in_channels = decoder_dims[i - 1] + encoder_dims[-(i + 1)]
            
            out_channels = decoder_dims[i]
            
            stage = nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(
                    decoder_dims[i - 1] if i > 0 else encoder_dims[-1],
                    decoder_dims[i - 1] if i > 0 else encoder_dims[-1],
                    kernel_size=2,
                    stride=2
                ) if i > 0 else nn.Identity(),
                
                'fusion_conv': nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ),
                
                'ucb': UrbanContextBlock(
                    channels=out_channels,
                    spatial_kernel=7,
                    reduction=16
                )
            })
            
            self.decoder_stages.append(stage)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Conv2d(decoder_dims[-1], num_classes, 1)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Decode features to segmentation map.
        
        Args:
            encoder_features: List of encoder features [F1, F2, F3, F4]
                             from shallow to deep
            return_intermediate: Whether to return intermediate decoder features
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Start from deepest features
        x = encoder_features[-1]
        
        intermediate_features = []
        
        # Progressive upsampling with skip connections
        for i, stage in enumerate(self.decoder_stages):
            # Upsample (except first stage)
            if i > 0:
                x = stage['upsample'](x)
                
                # Get corresponding skip connection
                skip_idx = -(i + 1)
                skip = encoder_features[skip_idx]
                
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x,
                        size=skip.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Concatenate with skip connection
                x = torch.cat([x, skip], dim=1)
            
            # Fusion convolution
            x = stage['fusion_conv'](x)
            
            # Urban Context Block refinement
            x = stage['ucb'](x)
            
            if return_intermediate:
                intermediate_features.append(x)
        
        # Final upsampling to original resolution
        x = self.final_upsample(x)
        
        # Classification
        logits = self.classifier(x)
        
        if return_intermediate:
            return logits, intermediate_features
        
        return logits


if __name__ == "__main__":
    print("Testing Urban Context Decoder components...")
    
    # Test Spatial Attention
    print("\n1. Testing Spatial Attention...")
    spatial_att = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 64, 64)
    att_map = spatial_att(x)
    print(f"Input: {x.shape} -> Attention Map: {att_map.shape}")
    
    # Test Channel Attention
    print("\n2. Testing Channel Attention...")
    channel_att = ChannelAttention(channels=64, reduction=16)
    att_map = channel_att(x)
    print(f"Input: {x.shape} -> Attention Map: {att_map.shape}")
    
    # Test Urban Context Block
    print("\n3. Testing Urban Context Block...")
    ucb = UrbanContextBlock(channels=64)
    x_enhanced = ucb(x)
    print(f"Input: {x.shape} -> Enhanced: {x_enhanced.shape}")
    
    # Test Urban Context Decoder
    print("\n4. Testing Urban Context Decoder...")
    decoder = UrbanContextDecoder(
        encoder_dims=[96, 192, 384, 768],
        decoder_dims=[384, 192, 96, 48],
        num_classes=6
    )
    
    # Simulate encoder features
    encoder_feats = [
        torch.randn(2, 96, 128, 128),   # Stage 1
        torch.randn(2, 192, 64, 64),    # Stage 2
        torch.randn(2, 384, 32, 32),    # Stage 3
        torch.randn(2, 768, 16, 16)     # Stage 4
    ]
    
    logits = decoder(encoder_feats)
    print(f"Encoder features: {[f.shape for f in encoder_feats]}")
    print(f"Output logits: {logits.shape}")
    print(f"Expected: (2, 6, 512, 512)")
    
    print("\n✓ All decoder tests passed!")
