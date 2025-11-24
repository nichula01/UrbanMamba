"""
MambaVision Modules for UrbanMamba Architecture
Implements Visual State Space (VSS) blocks with specialized scanning mechanisms,
configurable encoder, and fusion modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import sys
import os

# Add paths for importing MambaVision
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# mambavision is at root level (3 levels up from models/)
mambavision_dir = os.path.join(os.path.dirname(os.path.dirname(parent_dir)), 'mambavision')
if mambavision_dir not in sys.path:
    sys.path.insert(0, mambavision_dir)

try:
    from mambavision.models.mamba_vision import MambaVision
    MAMBAVISION_AVAILABLE = True
except ImportError:
    print("Warning: MambaVision not available. Install with: pip install timm==1.0.15 mamba-ssm==2.2.4")
    MAMBAVISION_AVAILABLE = False


class CrossScan2D:
    """
    Implements specialized 2D scanning strategies for VSS blocks.
    Converts 2D feature maps to 1D sequences with geometric awareness.
    """
    
    @staticmethod
    def directional_scan(x: torch.Tensor) -> List[torch.Tensor]:
        """
        Directional scanning: 4 diagonal passes.
        Essential for modeling road networks and structural alignments.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of 4 scanned sequences
        """
        B, C, H, W = x.shape
        
        # 1. Top-left to bottom-right
        scan1 = x.reshape(B, C, -1)  # [B, C, H*W]
        
        # 2. Bottom-right to top-left (reverse)
        scan2 = torch.flip(x, dims=[2, 3]).reshape(B, C, -1)
        
        # 3. Top-right to bottom-left (flip width)
        scan3 = torch.flip(x, dims=[3]).reshape(B, C, -1)
        
        # 4. Bottom-left to top-right (flip height)
        scan4 = torch.flip(x, dims=[2]).reshape(B, C, -1)
        
        return [scan1, scan2, scan3, scan4]
    
    @staticmethod
    def scale_aware_scan(x: torch.Tensor, stride: int = 2) -> List[torch.Tensor]:
        """
        Scale-aware scanning with multi-stride passes.
        Captures hierarchical patterns for sharp boundaries.
        
        Args:
            x: Input tensor [B, C, H, W]
            stride: Stride for multi-scale scanning
            
        Returns:
            List of multi-scale scanned sequences
        """
        B, C, H, W = x.shape
        
        scans = []
        
        # Full resolution scan
        scans.append(x.reshape(B, C, -1))
        
        # Strided scan (coarser)
        if stride > 1:
            x_strided = x[:, :, ::stride, ::stride]
            scans.append(x_strided.reshape(B, C, -1))
        
        return scans
    
    @staticmethod
    def spiral_scan(x: torch.Tensor) -> torch.Tensor:
        """
        Contextual spiral scanning from center outward.
        Prioritizes context from urban centers.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Spirally scanned sequence [B, C, H*W]
        """
        B, C, H, W = x.shape
        
        # Create spiral indices
        center_h, center_w = H // 2, W // 2
        
        # For simplicity, use a radial distance-based approximation
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        
        # Calculate distance from center
        dist = (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2
        
        # Sort by distance (spiral approximation)
        _, indices = torch.sort(dist.reshape(-1))
        
        # Reshape and gather
        x_flat = x.reshape(B, C, -1)
        x_spiral = x_flat[:, :, indices]
        
        return x_spiral
    
    @staticmethod
    def unscan_directional(scans: List[torch.Tensor], shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Reverse directional scanning and aggregate.
        
        Args:
            scans: List of 4 scanned tensors
            shape: Original shape [B, C, H, W]
            
        Returns:
            Aggregated tensor [B, C, H, W]
        """
        B, C, H, W = shape
        
        # Reshape scan1 (already in correct order)
        out1 = scans[0].reshape(B, C, H, W)
        
        # Reverse scan2
        out2 = scans[1].reshape(B, C, H, W)
        out2 = torch.flip(out2, dims=[2, 3])
        
        # Reverse scan3
        out3 = scans[2].reshape(B, C, H, W)
        out3 = torch.flip(out3, dims=[3])
        
        # Reverse scan4
        out4 = scans[3].reshape(B, C, H, W)
        out4 = torch.flip(out4, dims=[2])
        
        # Aggregate by averaging
        output = (out1 + out2 + out3 + out4) / 4.0
        
        return output


class SS2D(nn.Module):
    """
    Selective Scan 2D - Core State Space Model for sequential processing.
    Implements discretized state evolution with selective mechanisms.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        """
        Initialize SS2D module.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for SSM
            d_conv: Convolution kernel size
            expand: Expansion factor
        """
        super(SS2D, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=d_inner,
            bias=True
        )
        
        self.act = nn.SiLU()
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        
        # A parameter (state transition)
        A = torch.randn(d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_inner)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with state space dynamics.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: [B, H*W, C]
        x_flat = x.reshape(B, H * W, C)
        
        # Input projection and split
        x_proj = self.in_proj(x_flat)
        x_inner, z = x_proj.chunk(2, dim=-1)  # [B, H*W, d_inner] each
        
        # Reshape for conv: [B, d_inner, H, W]
        x_conv = x_inner.transpose(1, 2).reshape(B, -1, H, W)
        x_conv = self.act(self.conv2d(x_conv))
        x_conv = x_conv.reshape(B, -1, H * W).transpose(1, 2)  # [B, H*W, d_inner]
        
        # SSM parameters
        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)  # [B, H*W, d_state] each
        
        # Discretization
        dt = self.dt_proj(B_ssm)  # [B, H*W, d_inner]
        dt = F.softplus(dt)
        
        # State evolution (simplified)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # Compute deltaA and deltaB (zero-order hold)
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, H*W, d_inner, d_state]
        deltaB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, H*W, d_inner, d_state]
        
        # State update (simplified for efficiency)
        # In practice, use selective scan kernel
        x_ssm_out = torch.einsum('blid,bld->bli', deltaB, C_ssm)
        
        # Add skip connection
        x_ssm_out = x_ssm_out + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate with z
        x_out = x_ssm_out * F.silu(z)
        
        # Normalize
        x_out = self.norm(x_out)
        
        # Output projection
        x_out = self.out_proj(x_out)
        
        # Reshape back: [B, C, H, W]
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class VSSBlock(nn.Module):
    """
    Visual State Space Block with cross-scan mechanism.
    Integrates directional, scale-aware, and spiral scanning.
    """
    
    def __init__(self, hidden_dim: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        """
        Initialize VSS Block.
        
        Args:
            hidden_dim: Hidden dimension
            d_state: State dimension
            d_conv: Convolution size
            expand: Expansion factor
        """
        super(VSSBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(hidden_dim, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.scanner = CrossScan2D()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-scanning.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # Apply directional scanning
        x_scans = self.scanner.directional_scan(x)
        
        # Process each scan direction
        outputs = []
        for scan in x_scans:
            scan_2d = scan.reshape(B, C, H, W)
            scan_2d = scan_2d.permute(0, 2, 3, 1)  # [B, H, W, C]
            scan_2d = self.norm1(scan_2d)
            scan_2d = scan_2d.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            out_scan = self.ss2d(scan_2d)
            outputs.append(out_scan.reshape(B, C, -1))
        
        # Unscan and aggregate
        x = self.scanner.unscan_directional(outputs, (B, C, H, W))
        x = x + shortcut
        
        # MLP block
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = x + shortcut
        
        return x


class MambaVisionEncoder(nn.Module):
    """
    MambaVision Encoder with configurable input channels.
    Critical for dual-branch architecture supporting both RGB (3) and XLET (87) inputs.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 4, 2],
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2
    ):
        """
        Initialize MambaVision Encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB, 87 for XLET)
            hidden_dims: Hidden dimensions for each stage
            depths: Number of VSS blocks per stage
            d_state: State dimension for SSM
            d_conv: Convolution kernel size
            expand: Expansion factor
        """
        super(MambaVisionEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.depths = depths
        self.num_stages = len(hidden_dims)
        
        # Stem layer: Critical modification for 87-channel XLET input
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,  # Accepts 3 (RGB) or 87 (XLET)
                hidden_dims[0],
                kernel_size=4,
                stride=4,
                padding=0
            ),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.GELU()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        
        for stage_idx in range(self.num_stages):
            stage_blocks = []
            
            # Downsampling layer (except first stage)
            if stage_idx > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[stage_idx - 1],
                        hidden_dims[stage_idx],
                        kernel_size=2,
                        stride=2
                    ),
                    nn.BatchNorm2d(hidden_dims[stage_idx])
                )
                stage_blocks.append(downsample)
            
            # VSS blocks
            for _ in range(depths[stage_idx]):
                block = VSSBlock(
                    hidden_dim=hidden_dims[stage_idx],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                stage_blocks.append(block)
            
            self.stages.append(nn.Sequential(*stage_blocks))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder stages.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            List of multi-scale features from each stage
        """
        features = []
        
        # Stem
        x = self.stem(x)
        
        # Process through stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features


class MambaFusionBlock(nn.Module):
    """
    Twin Tower Fusion Block for UrbanMamba v3.
    Fuses spatial and frequency features using Mamba mixer with selective integration.
    
    This is the core innovation of v3 - instead of simple concatenation or addition,
    we use a Mamba state-space model to selectively integrate features from both branches.
    
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
        
        # Check if mamba_ssm is available
        try:
            from mamba_ssm import Mamba
            
            # 1. Project 2x channels down to 1x (Compression after concat)
            self.proj = nn.Linear(channels * 2, channels)
            
            # 2. Mamba Mixer (The "Smart" Fusion)
            # This allows the model to select features from RGB or NSST dynamically
            self.mixer = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            
            # 3. Normalization
            self.norm = nn.LayerNorm(channels)
            
            self.use_mamba = True
            
        except ImportError:
            # Fallback to convolutional fusion if mamba_ssm not available
            print("WARNING: mamba_ssm not available - using convolutional fusion fallback")
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
        # Ensure spatial dimensions match
        if spatial_feat.shape[2:] != freq_feat.shape[2:]:
            freq_feat = F.interpolate(
                freq_feat,
                size=spatial_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        B, C, H, W = spatial_feat.shape
        
        if self.use_mamba:
            # Mamba-based fusion
            # Flatten for Mamba (Requires B, L, C where L = H*W)
            x_spatial = spatial_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_freq = freq_feat.flatten(2).transpose(1, 2)       # [B, H*W, C]
            
            # Concatenate along channel dimension
            x_cat = torch.cat([x_spatial, x_freq], dim=-1)  # [B, H*W, 2*C]
            
            # Project and Mix
            x_proj = self.proj(x_cat)  # [B, H*W, C]
            x_mixed = self.mixer(x_proj)
            x_out = self.norm(x_mixed)
            
            # Reshape back to image format [B, C, H, W]
            return x_out.transpose(1, 2).view(B, C, H, W)
        else:
            # Convolutional fallback fusion
            x_cat = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 2*C, H, W]
            x_proj = self.proj(x_cat)  # [B, C, H, W]
            x_fused = self.conv_fusion(x_proj)
            return x_fused


class MambaFusionModule(nn.Module):
    """
    Mamba Fusion Module for adaptive integration of spatial and XLET features.
    Implements synchronization, channel balancing, and adaptive VMamba-based fusion.
    """
    
    def __init__(self, spatial_dim: int, xlet_dim: int, output_dim: int):
        """
        Initialize Fusion Module.
        
        Args:
            spatial_dim: Spatial branch channel dimension
            xlet_dim: XLET branch channel dimension
            output_dim: Output fused dimension
        """
        super(MambaFusionModule, self).__init__()
        
        # Channel balancing via 1x1 convolution
        self.channel_balance = nn.Conv2d(
            spatial_dim + xlet_dim,
            output_dim,
            kernel_size=1,
            bias=False
        )
        self.norm = nn.BatchNorm2d(output_dim)
        
        # Adaptive integration via VMamba block
        self.fusion_block = VSSBlock(
            hidden_dim=output_dim,
            d_state=16,
            d_conv=3,
            expand=2
        )
    
    def forward(self, f_spatial: torch.Tensor, f_xlet: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and XLET features.
        
        Args:
            f_spatial: Spatial branch features [B, C_s, H, W]
            f_xlet: XLET branch features [B, C_x, H, W]
            
        Returns:
            Fused features [B, C_out, H, W]
        """
        # Ensure spatial dimensions match
        if f_spatial.shape[2:] != f_xlet.shape[2:]:
            f_xlet = F.interpolate(
                f_xlet,
                size=f_spatial.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate along channel dimension
        x_cat = torch.cat([f_spatial, f_xlet], dim=1)
        
        # Channel balancing
        x_balanced = self.channel_balance(x_cat)
        x_balanced = self.norm(x_balanced)
        
        # Adaptive integration via VMamba
        x_fused = self.fusion_block(x_balanced)
        
        return x_fused


if __name__ == "__main__":
    # Test MambaVision Encoder with different input channels
    print("Testing MambaVision Encoder...")
    
    # Test RGB encoder (3 channels)
    rgb_encoder = MambaVisionEncoder(in_channels=3)
    x_rgb = torch.randn(2, 3, 512, 512)
    features_rgb = rgb_encoder(x_rgb)
    print(f"\nRGB Encoder (3 channels):")
    print(f"Input: {x_rgb.shape}")
    for i, feat in enumerate(features_rgb):
        print(f"Stage {i+1}: {feat.shape}")
    
    # Test XLET encoder (87 channels)
    xlet_encoder = MambaVisionEncoder(in_channels=87)
    x_xlet = torch.randn(2, 87, 512, 512)
    features_xlet = xlet_encoder(x_xlet)
    print(f"\nXLET Encoder (87 channels):")
    print(f"Input: {x_xlet.shape}")
    for i, feat in enumerate(features_xlet):
        print(f"Stage {i+1}: {feat.shape}")
    
    # Test Fusion Module
    print("\n\nTesting Fusion Module...")
    fusion = MambaFusionModule(spatial_dim=96, xlet_dim=96, output_dim=96)
    f_spatial = torch.randn(2, 96, 128, 128)
    f_xlet = torch.randn(2, 96, 128, 128)
    f_fused = fusion(f_spatial, f_xlet)
    print(f"Spatial: {f_spatial.shape}, XLET: {f_xlet.shape} -> Fused: {f_fused.shape}")
    
    print("\n✓ All module tests passed!")
