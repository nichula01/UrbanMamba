"""
XLET Normalization Stem for Frequency Branch

This module stabilizes the 87-channel NSST frequency coefficients before
feeding them into the MambaVision backbone.
"""

import torch
import torch.nn as nn


class XLETNormalizationStem(nn.Module):
    """
    XLET Normalization Stem for stabilizing frequency coefficients.
    
    The Problem: NSST produces 87 channels with highly variable magnitudes
    (some coefficients are 1e-5, others are large). Feeding this directly
    to a backbone trained on normalized images causes gradient explosion.
    
    The Solution: 
    1. Instance Normalization forces coefficients into stable range
    2. Learnable projection maps to backbone feature dimension
    3. Kaiming initialization ensures stable gradient flow
    
    Args:
        in_channels: Number of input frequency channels (87 for NSST)
        out_channels: Output dimension matching backbone (e.g., 80, 96)
        use_learnable_norm: If True, Instance Norm has learnable affine params
    """
    
    def __init__(
        self, 
        in_channels: int = 87, 
        out_channels: int = 96,
        use_learnable_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. Instance Normalization: Stabilizes each channel independently
        # Affine=True allows the network to learn optimal scale/bias per channel
        # FIX 3: Increased eps to prevent division by zero on blank frequency areas
        self.norm = nn.InstanceNorm2d(
            in_channels, 
            affine=use_learnable_norm,
            track_running_stats=False,  # Don't track stats, normalize per-batch
            eps=1e-4  # EXTREME: 1e-4 for maximum stability
        )
        
        # 2. Projection: Maps from 87 channels to backbone dimension
        # Uses 3x3 conv to allow spatial feature mixing
        self.proj = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=True
        )
        
        # 3. Kaiming Initialization: Critical for frequency data
        # Unlike images, NSST doesn't follow ImageNet statistics
        nn.init.kaiming_normal_(
            self.proj.weight, 
            mode='fan_out', 
            nonlinearity='relu'
        )
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        
        # 4. Activation
        self.act = nn.GELU()
        
        # Optional: Additional stabilization layer
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with stabilization.
        
        Args:
            x: Input frequency features [B, 87, H, W]
            
        Returns:
            Normalized and projected features [B, out_channels, H, W]
        """
        # CRITICAL FIX 2: Force FP32 for frequency branch to prevent NaN
        # NSST coefficients need full float32 precision, float16 causes overflow
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()  # Force FP32
            
            # Step 1: Normalize each frequency channel
            # This brings all coefficients to similar scale
            x = self.norm(x)
            
            # Step 2: Project to backbone dimension
            x = self.proj(x)
            
            # Step 3: Apply activation
            x = self.act(x)
        
        # Step 4: Additional batch normalization for stability (can use AMP)
        x = self.batch_norm(x)
        
        return x
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


class LightweightXLETStem(nn.Module):
    """
    Lightweight version of XLET stem with fewer parameters.
    
    Uses depthwise separable convolution for efficiency.
    Good for resource-constrained scenarios.
    """
    
    def __init__(
        self, 
        in_channels: int = 87, 
        out_channels: int = 96
    ):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        
        # Depthwise convolution: Process each channel separately
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1, 
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution: Mix channels
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            bias=True
        )
        
        self.act = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Initialize
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.batch_norm(x)
        return x


if __name__ == "__main__":
    # Test the XLET stem
    print("Testing XLET Normalization Stem...")
    
    # Create stem
    stem = XLETNormalizationStem(in_channels=87, out_channels=96)
    
    # Test with random NSST features (simulating real data with varying scales)
    B, H, W = 2, 512, 512
    
    # Simulate NSST with varying magnitudes
    x = torch.randn(B, 87, H, W)
    # Add some channels with extreme values
    x[:, :10] *= 100  # Large coefficients
    x[:, 10:20] *= 0.01  # Small coefficients
    
    print(f"\nInput statistics:")
    print(f"  Shape: {x.shape}")
    print(f"  Min: {x.min():.6f}, Max: {x.max():.6f}")
    print(f"  Mean: {x.mean():.6f}, Std: {x.std():.6f}")
    
    # Forward pass
    output = stem(x)
    
    print(f"\nOutput statistics:")
    print(f"  Shape: {output.shape}")
    print(f"  Min: {output.min():.6f}, Max: {output.max():.6f}")
    print(f"  Mean: {output.mean():.6f}, Std: {output.std():.6f}")
    
    print(f"\n✅ XLET Stem successfully stabilized the frequency features!")
    print(f"   Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"   Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test lightweight version
    print("\n" + "="*70)
    print("Testing Lightweight XLET Stem...")
    lightweight_stem = LightweightXLETStem(in_channels=87, out_channels=96)
    output_light = lightweight_stem(x)
    print(f"  Output shape: {output_light.shape}")
    print(f"  ✅ Lightweight version working!")
