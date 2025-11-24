"""
NSST (Non-Subsampled Shearlet Transform) for frequency feature extraction.
This extracts 87 frequency subbands from RGB images for the frequency encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class NSSTDecomposition(nn.Module):
    """
    Non-Subsampled Shearlet Transform (NSST) for multi-scale, multi-directional frequency analysis.
    
    This module decomposes an RGB image into frequency subbands across multiple scales and directions.
    The output is a stacked tensor with all subbands concatenated along the channel dimension.
    
    Architecture:
    - 3 scales with directions [2, 3, 4] = [4, 8, 16] directional subbands per scale
    - Low-frequency residual at each scale
    - Total: (4 + 8 + 16) + 3*1 = 28 + 3 = 31 subbands per RGB channel
    - For RGB (3 channels): 31 * 3 = 93 channels... but we use (4+8+16+1)*3 = 87 channels
    
    Actually for 87 channels:
    Scale 1: 2^2 = 4 directions + 1 low-freq = 5 bands
    Scale 2: 2^3 = 8 directions + 1 low-freq = 9 bands
    Scale 3: 2^4 = 16 directions + 1 low-freq = 17 bands
    Total per channel: 5 + 9 + 17 = 31? No...
    
    Let me recalculate: directions_profile=[2, 3, 4] means:
    - Scale 1: 2 directions
    - Scale 2: 3 directions  
    - Scale 3: 4 directions
    Total per scale: 2 + 3 + 4 = 9 directional + 3 low-freq = 12 per channel
    For 3 channels: 12 * 3 = 36... still not 87
    
    Actually, the profile means 2^2, 2^3, 2^4 directional filters:
    - Scale 1: 4 directions
    - Scale 2: 8 directions
    - Scale 3: 16 directions
    Plus 1 low-frequency per scale = 3
    Total: (4 + 8 + 16 + 1) * 3 channels = 29 * 3 = 87 ✓
    
    Args:
        scales: Number of decomposition scales (default: 3)
        directions_profile: List of direction exponents [2, 3, 4] for each scale
    """
    
    def __init__(
        self,
        scales: int = 3,
        directions_profile: List[int] = [2, 3, 4]
    ):
        super().__init__()
        self.scales = scales
        self.directions_profile = directions_profile
        
        # Calculate total number of subbands
        # For each scale: 2^d directions, where d is from directions_profile
        # Plus 1 low-frequency component at the end
        self.num_subbands_per_channel = sum(2**d for d in directions_profile) + 1
        self.total_channels = self.num_subbands_per_channel * 3  # RGB
        
        assert self.total_channels == 87, f"Expected 87 channels, got {self.total_channels}"
        
        # Initialize shearlet filters (learnable or fixed)
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Initialize shearlet filters for each scale and direction."""
        self.filters = nn.ParameterDict()
        
        for scale_idx, num_dir_exp in enumerate(self.directions_profile):
            num_directions = 2 ** num_dir_exp
            
            # Create directional filters for this scale
            # Filter size increases with scale: [11, 15, 19]
            filter_size = 11 + 4 * scale_idx
            
            for dir_idx in range(num_directions):
                # Angle for this direction
                angle = np.pi * dir_idx / num_directions
                
                # Create oriented Gabor-like filter
                filter_kernel = self._create_shearlet_filter(filter_size, angle, scale_idx)
                
                # Register as parameter (learnable)
                param_name = f"scale{scale_idx}_dir{dir_idx}"
                self.filters[param_name] = nn.Parameter(
                    torch.from_numpy(filter_kernel).float().unsqueeze(0).unsqueeze(0),
                    requires_grad=False  # Keep fixed for stability
                )
    
    def _create_shearlet_filter(self, size: int, angle: float, scale: int) -> np.ndarray:
        """
        Create a shearlet (directional) filter at given angle and scale.
        
        Args:
            size: Filter kernel size
            angle: Orientation angle in radians
            scale: Scale index (0, 1, 2)
            
        Returns:
            2D filter kernel [size, size]
        """
        # Create coordinate grids
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        Y_rot = -X * np.sin(angle) + Y * np.cos(angle)
        
        # Frequency for this scale (higher scale = lower frequency)
        freq = 0.3 / (2 ** scale)
        
        # Directional bandwidth (controls direction selectivity)
        dir_bandwidth = 2.0
        
        # Create Gabor-like filter
        gaussian_envelope = np.exp(-(X_rot**2 + (dir_bandwidth * Y_rot)**2) / (2 * (size/4)**2))
        sinusoid = np.cos(2 * np.pi * freq * X_rot)
        
        filter_kernel = gaussian_envelope * sinusoid
        
        # Normalize
        filter_kernel = filter_kernel / (np.abs(filter_kernel).sum() + 1e-8)
        
        return filter_kernel
    
    def _apply_filter(self, image: torch.Tensor, filter_kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply a single filter to an image (works on single channel).
        
        Args:
            image: Input [B, 1, H, W]
            filter_kernel: Filter [1, 1, K, K]
            
        Returns:
            Filtered output [B, 1, H, W]
        """
        # Get padding size
        pad_size = filter_kernel.shape[-1] // 2
        
        # Apply convolution with padding
        filtered = F.conv2d(
            image,
            filter_kernel.to(image.device),
            padding=pad_size
        )
        
        return filtered
    
    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Decompose RGB image into NSST frequency subbands.
        
        Args:
            rgb_image: Input RGB image [B, 3, H, W]
            
        Returns:
            NSST features [B, 87, H, W] - stacked frequency subbands
        """
        B, C, H, W = rgb_image.shape
        assert C == 3, f"Expected RGB image with 3 channels, got {C}"
        
        all_subbands = []
        
        # Process each RGB channel separately
        for channel_idx in range(3):
            channel_image = rgb_image[:, channel_idx:channel_idx+1, :, :]  # [B, 1, H, W]
            
            channel_subbands = []
            
            # Apply filters for each scale and direction
            for scale_idx, num_dir_exp in enumerate(self.directions_profile):
                num_directions = 2 ** num_dir_exp
                
                for dir_idx in range(num_directions):
                    param_name = f"scale{scale_idx}_dir{dir_idx}"
                    filter_kernel = self.filters[param_name]
                    
                    # Apply filter
                    subband = self._apply_filter(channel_image, filter_kernel)
                    channel_subbands.append(subband)
            
            # Add low-frequency component (downsampled then upsampled)
            low_freq = F.avg_pool2d(channel_image, kernel_size=4, stride=4)
            low_freq = F.interpolate(low_freq, size=(H, W), mode='bilinear', align_corners=False)
            channel_subbands.append(low_freq)
            
            # Concatenate all subbands for this channel
            all_subbands.extend(channel_subbands)
        
        # Stack all subbands: [B, 87, H, W]
        nsst_features = torch.cat(all_subbands, dim=1)
        
        assert nsst_features.shape[1] == 87, f"Expected 87 channels, got {nsst_features.shape[1]}"
        
        return nsst_features
    
    def get_num_channels(self) -> int:
        """Return the number of output channels (87 for RGB input)."""
        return self.total_channels


if __name__ == "__main__":
    # Test NSST decomposition
    print("Testing NSST Decomposition...")
    
    nsst = NSSTDecomposition(scales=3, directions_profile=[2, 3, 4])
    print(f"✓ NSST created: {nsst.get_num_channels()} output channels")
    
    # Test forward pass
    dummy_rgb = torch.randn(2, 3, 256, 256)
    print(f"Input shape: {dummy_rgb.shape}")
    
    with torch.no_grad():
        nsst_features = nsst(dummy_rgb)
    
    print(f"Output shape: {nsst_features.shape}")
    print(f"✓ NSST decomposition successful!")
    
    # Verify channel count
    assert nsst_features.shape == (2, 87, 256, 256), "Unexpected output shape"
    print("✓ All tests passed!")
