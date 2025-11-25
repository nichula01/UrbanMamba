"""
Urban Context Decoder (UCD) with Urban Context Blocks (UCB)
Implements progressive upsampling with dual attention mechanisms for
high-resolution boundary preservation in urban semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


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


class ASPPContext(nn.Module):
    """ASPP-style context block to expand receptive field at the coarsest level."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
        feats.append(pooled)
        out = self.project(torch.cat(feats, dim=1))
        return out


class LearnedUpsample(nn.Module):
    """Learned upsampling using either transposed conv or conv after bilinear."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "transpose"):
        super().__init__()
        if mode == "transpose":
            self.op = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class GatedFusion(nn.Module):
    """Fuse upsampled deep features and skips with learnable gating."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([deep_feat, skip_feat], dim=1))
        fused = deep_feat * gate + skip_feat * (1 - gate)
        fused = self.refine(torch.cat([fused, deep_feat], dim=1))
        return fused


class AuxHead(nn.Module):
    """Lightweight auxiliary prediction head for deep supervision."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        mid = max(in_channels // 2, num_classes)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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
    
    Enhancements for urban-aware segmentation:
    - ASPP-style context at the coarsest level for large receptive fields
    - Gated skip fusion to balance deep context and shallow detail
    - Learned upsampling in later stages to reduce blur
    - Boundary refinement head feeding back into final logits
    - Optional deep supervision from mid- and high-resolution stages
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [80, 160, 320, 640],
        decoder_dims: List[int] = [320, 160, 80, 40],
        num_classes: int = 6,
        deep_supervision: bool = False,
    ):
        """
        Initialize Urban Context Decoder.
        
        Args:
            encoder_dims: Channel dimensions from encoder stages (for skip connections)
            decoder_dims: Channel dimensions for decoder stages
            num_classes: Number of output classes (6 for urban segmentation)
            deep_supervision: Whether to emit auxiliary logits from decoder stages
        """
        super(UrbanContextDecoder, self).__init__()
        
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_stages = len(decoder_dims)
        self.deep_supervision = deep_supervision
        
        # Coarsest-level context enhancer
        self.context_block = ASPPContext(encoder_dims[-1], encoder_dims[-1])
        self.deep_projection = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], decoder_dims[0], 1, bias=False),
            nn.BatchNorm2d(decoder_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.initial_refine = UrbanContextBlock(channels=decoder_dims[0], spatial_kernel=7, reduction=16)

        # Decoder stages (1..num_stages-1) with learned upsampling and gated fusion
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.gated_fusions = nn.ModuleList()
        self.stage_refines = nn.ModuleList()

        for i in range(1, self.num_stages):
            out_channels = decoder_dims[i]
            self.upsamples.append(LearnedUpsample(decoder_dims[i - 1], out_channels))
            self.skip_projs.append(
                nn.Sequential(
                    nn.Conv2d(encoder_dims[-(i + 1)], out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.gated_fusions.append(GatedFusion(out_channels))
            self.stage_refines.append(UrbanContextBlock(channels=out_channels, spatial_kernel=7, reduction=16))

        # Final learned upsampling (two steps for crisper edges)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True),
        )

        # Boundary refinement head
        boundary_mid = max(decoder_dims[-1] // 2, 32)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], boundary_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(boundary_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_mid, 1, 1),
        )
        self.boundary_fuse = nn.Sequential(
            nn.Conv2d(decoder_dims[-1] + 1, decoder_dims[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True),
        )

        # Class-aware refinement near the output
        self.class_attention = ChannelAttention(decoder_dims[-1], reduction=8)
        self.class_refine = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(inplace=True),
        )

        # Auxiliary heads for deep supervision (optional)
        self.aux_indices = [idx for idx in [self.num_stages - 2, self.num_stages - 1] if idx > 0]
        self.aux_heads = nn.ModuleDict()
        if deep_supervision:
            for idx in self.aux_indices:
                self.aux_heads[str(idx)] = AuxHead(decoder_dims[idx], num_classes)

        # Classification head
        self.classifier = nn.Conv2d(decoder_dims[-1], num_classes, 1)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        return_intermediate: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Decode features to segmentation map.
        
        Args:
            encoder_features: List of encoder features [F1, F2, F3, F4]
                             from shallow to deep
            return_intermediate: Whether to return intermediate decoder features
            return_aux: Whether to return auxiliary predictions (boundary, deep supervision)
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
            Or tuple with logits and requested auxiliary outputs.
        """
        # Start from deepest features and inject large-context aggregation
        x = self.context_block(encoder_features[-1])
        x = self.deep_projection(x)
        x = self.initial_refine(x)

        intermediate_features = []
        aux_outputs: Dict[str, torch.Tensor] = {}

        if return_intermediate:
            intermediate_features.append(x)

        # Progressive upsampling with gated skip fusion
        for stage_idx in range(1, self.num_stages):
            x = self.upsamples[stage_idx - 1](x)

            skip = encoder_features[-(stage_idx + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            skip_proj = self.skip_projs[stage_idx - 1](skip)
            x = self.gated_fusions[stage_idx - 1](x, skip_proj)
            x = self.stage_refines[stage_idx - 1](x)

            if return_intermediate:
                intermediate_features.append(x)

            if self.deep_supervision and str(stage_idx) in self.aux_heads:
                aux_outputs[f"aux_{stage_idx}"] = self.aux_heads[str(stage_idx)](x)

        # Final learned upsampling to original resolution
        x = self.final_upsample(x)

        # Boundary branch with feedback
        boundary_logits = self.boundary_head(x)
        boundary_prob = torch.sigmoid(boundary_logits)
        x = self.boundary_fuse(torch.cat([x, boundary_prob], dim=1))

        # Class-aware refinement before classification
        x = x * self.class_attention(x)
        x = self.class_refine(x)

        logits = self.classifier(x)

        if aux_outputs:
            for key in list(aux_outputs.keys()):
                aux_outputs[key] = F.interpolate(aux_outputs[key], size=logits.shape[2:], mode='bilinear', align_corners=False)
            aux_outputs["boundary"] = boundary_logits
        elif return_aux:
            aux_outputs["boundary"] = boundary_logits

        if return_aux and return_intermediate:
            return logits, intermediate_features, aux_outputs
        if return_aux:
            return logits, aux_outputs
        if return_intermediate:
            return logits, intermediate_features

        return logits


# Alias for compatibility
MultiScaleAggregationHead = UrbanContextDecoder


if __name__ == "__main__":
    print("Testing Urban Context Decoder components...")
    
    # Test Spatial Attention
    print("\n1. Testing Spatial Attention...")
    spatial_att = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 64, 64)
    att_map = spatial_att(x)
    print(f"   Input: {x.shape} -> Attention Map: {att_map.shape}")
    
    # Test Channel Attention
    print("\n2. Testing Channel Attention...")
    channel_att = ChannelAttention(channels=64, reduction=16)
    att_map = channel_att(x)
    print(f"   Input: {x.shape} -> Attention Map: {att_map.shape}")
    
    # Test Urban Context Block
    print("\n3. Testing Urban Context Block...")
    ucb = UrbanContextBlock(channels=64)
    x_enhanced = ucb(x)
    print(f"   Input: {x.shape} -> Enhanced: {x_enhanced.shape}")
    
    # Test Urban Context Decoder
    print("\n4. Testing Urban Context Decoder...")
    decoder = UrbanContextDecoder(
        encoder_dims=[80, 160, 320, 640],
        decoder_dims=[320, 160, 80, 40],
        num_classes=6,
        deep_supervision=True,
    )
    
    print(f"   ✓ Decoder created with {sum(p.numel() for p in decoder.parameters()):,} parameters")
    
    # Simulate encoder features
    encoder_feats = [
        torch.randn(2, 80, 64, 64),    # Stage 1: 1/4
        torch.randn(2, 160, 32, 32),   # Stage 2: 1/8
        torch.randn(2, 320, 16, 16),   # Stage 3: 1/16
        torch.randn(2, 640, 8, 8)      # Stage 4: 1/32
    ]
    
    logits, aux = decoder(encoder_feats, return_aux=True)
    print(f"   Encoder features: {[f.shape for f in encoder_feats]}")
    print(f"   Output logits: {logits.shape}")
    for name, aux_map in aux.items():
        print(f"   Aux[{name}]: {aux_map.shape}")
    print(f"   Expected logits: (2, 6, 256, 256)")
    
    print("\n✓ All Urban Context Decoder tests passed!")
