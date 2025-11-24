"""
XLET-NSST Feature Extraction Module
Implements Non-Subsampled Shearlet Transform (NSST) for frequency-domain feature extraction.

This module replaces the Haar DWT with NSST decomposition, generating 87 feature channels
from 3-channel RGB input using 3 scales with directional decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class NSSTDecomposition(nn.Module):
    """
    Non-Subsampled Shearlet Transform (NSST) Decomposition.
    
    Decomposes input image into multi-scale, multi-directional subbands while
    preserving spatial resolution (non-subsampled).
    
    For 3-scale decomposition with profile [1, 2, 3]:
    - Scale 1: 1 lowpass + 2^1 = 2 directional bands = 3 total
    - Scale 2: 1 lowpass + 2^2 = 4 directional bands = 5 total  
    - Scale 3: 1 lowpass + 2^3 = 8 directional bands = 9 total
    - Total per channel: 3 + 5 + 9 = 17... but we concatenate all intermediate results
    
    Actually: 1 LP + (2 + 4 + 8) = 1 + 14 = 15 per scale cascade = 29 total per channel
    For RGB: 29 × 3 = 87 channels
    """
    
    def __init__(self, scales: int = 3, directions_profile: List[int] = None):
        """
        Initialize NSST Decomposition.
        
        Args:
            scales: Number of decomposition scales (default: 3)
            directions_profile: List of directional decomposition levels per scale.
                              To get 87 channels: use [2, 3, 4] for 3 scales
                              This gives 29 per channel × 3 = 87 total
        """
        super(NSSTDecomposition, self).__init__()
        
        self.scales = scales
        self.directions_profile = directions_profile or [2, 3, 4]
        
        # Calculate total channels: For each scale, we get 1 LP + 2^level directional bands
        # We concatenate all: LP_final + all directional bands from all scales
        self.channels_per_input = self._calculate_total_channels()
        
        # Initialize filter banks for Non-Subsampled Laplacian Pyramid
        self._init_laplacian_filters()
        
        # Initialize Directional Filter Banks for each scale
        self._init_directional_filters()
    
    def _calculate_total_channels(self) -> int:
        """Calculate total output channels per input channel."""
        # For profile [2, 3, 4]:
        # - 1 final lowpass
        # - 2^2 = 4 directional bands from scale 1
        # - 2^3 = 8 directional bands from scale 2
        # - 2^4 = 16 directional bands from scale 3
        # Total = 1 + 4 + 8 + 16 = 29 per channel × 3 channels = 87 total
        total = 1  # Final lowpass
        for level in self.directions_profile:
            total += 2 ** level  # Directional bands per scale
        return total
    
    def _init_laplacian_filters(self):
        """Initialize Non-Subsampled Laplacian Pyramid filters."""
        # Create low-pass filter (approximation)
        # Using a simple Gaussian-like kernel
        kernel_size = 5
        sigma = 1.0
        
        # Create 1D Gaussian kernel
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel_1d = gauss / np.sum(gauss)
        
        # Create 2D kernel
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / np.sum(kernel_2d)
        
        # Convert to torch tensor
        self.lowpass_kernel = torch.FloatTensor(kernel_2d).unsqueeze(0).unsqueeze(0)
        
        # Register as buffer (not trainable)
        self.register_buffer('lp_filter', self.lowpass_kernel)
    
    def _init_directional_filters(self):
        """Initialize Directional Filter Banks (DFB) for each scale."""
        # For simplicity, we use oriented Gabor-like filters
        # In production, use proper shearlet filters
        
        self.directional_filters = nn.ModuleDict()
        
        for scale_idx, n_dirs_level in enumerate(self.directions_profile):
            n_directions = 2 ** n_dirs_level
            filters = []
            
            for dir_idx in range(n_directions):
                # Create oriented filter
                angle = np.pi * dir_idx / n_directions
                filter_kernel = self._create_oriented_filter(angle, scale=scale_idx + 1)
                filters.append(filter_kernel)
            
            # Stack all directional filters for this scale
            filters_tensor = torch.stack(filters, dim=0)  # [n_directions, 1, H, W]
            self.register_buffer(f'dir_filters_scale_{scale_idx}', filters_tensor)
    
    def _create_oriented_filter(self, angle: float, scale: int = 1) -> torch.Tensor:
        """
        Create an oriented directional filter (simplified shearlet).
        
        Args:
            angle: Orientation angle in radians
            scale: Scale factor
            
        Returns:
            Oriented filter kernel [1, 1, H, W]
        """
        size = 15  # Filter size
        sigma_x = 2.0 * scale
        sigma_y = 0.5 * scale
        
        # Create coordinate grid
        x = np.arange(-size // 2 + 1, size // 2 + 1)
        y = np.arange(-size // 2 + 1, size // 2 + 1)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(angle) - Y * np.sin(angle)
        Y_rot = X * np.sin(angle) + Y * np.cos(angle)
        
        # Create oriented Gaussian (simplified shearlet approximation)
        kernel = np.exp(-0.5 * (X_rot**2 / sigma_x**2 + Y_rot**2 / sigma_y**2))
        
        # Add directional derivative for edge detection
        kernel = kernel * X_rot / sigma_x**2
        
        # Normalize
        kernel = kernel - np.mean(kernel)
        kernel_norm = np.linalg.norm(kernel)
        if kernel_norm > 1e-6:
            kernel = kernel / kernel_norm
        
        # Return as [1, 1, H, W]
        return torch.FloatTensor(kernel).view(1, 1, size, size)
    
    def _apply_filter(self, x: torch.Tensor, filter_kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply a filter to input with proper padding to maintain spatial dimensions.
        
        Args:
            x: Input tensor [B, C, H, W]
            filter_kernel: Filter kernel [1, 1, Kh, Kw]
            
        Returns:
            Filtered tensor [B, C, H, W]
        """
        # Ensure filter is on same device and correct shape
        filter_kernel = filter_kernel.to(x.device)
        
        # Ensure filter is 4D [1, 1, Kh, Kw]
        if filter_kernel.dim() == 2:
            filter_kernel = filter_kernel.unsqueeze(0).unsqueeze(0)
        elif filter_kernel.dim() == 5:
            filter_kernel = filter_kernel.squeeze(0)
        
        # Calculate padding to maintain size
        pad_h = filter_kernel.shape[2] // 2
        pad_w = filter_kernel.shape[3] // 2
        
        # Pad input
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # Get dimensions
        C = x.shape[1]
        kh, kw = filter_kernel.shape[2], filter_kernel.shape[3]
        
        # Expand filter for all channels: [C, 1, Kh, Kw]
        filter_expanded = filter_kernel.expand(C, 1, kh, kw)
        
        # Apply depthwise convolution
        output = F.conv2d(x_padded, filter_expanded, groups=C)
        
        return output
    
    def _nonsubsampled_laplacian_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute Non-Subsampled Laplacian Pyramid.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of [lowpass, bandpass_1, bandpass_2, ..., bandpass_n]
        """
        pyramid = []
        current = x
        
        for scale in range(self.scales):
            # Apply lowpass filter (no subsampling)
            lowpass = self._apply_filter(current, self.lp_filter)
            
            # Bandpass = current - lowpass
            bandpass = current - lowpass
            pyramid.append(bandpass)
            
            # Continue with lowpass for next scale
            current = lowpass
        
        # Add final lowpass residual
        pyramid.append(current)
        
        return pyramid
    
    def _directional_decomposition(self, bandpass: torch.Tensor, scale_idx: int) -> List[torch.Tensor]:
        """
        Apply directional decomposition to a bandpass component.
        
        Args:
            bandpass: Bandpass tensor [B, C, H, W]
            scale_idx: Scale index
            
        Returns:
            List of directional subbands
        """
        dir_filters = getattr(self, f'dir_filters_scale_{scale_idx}')
        n_directions = dir_filters.shape[0]
        
        directional_subbands = []
        for dir_idx in range(n_directions):
            dir_filter = dir_filters[dir_idx:dir_idx+1]  # [1, 1, H, W]
            subband = self._apply_filter(bandpass, dir_filter)
            directional_subbands.append(subband)
        
        return directional_subbands
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform NSST decomposition on input image.
        
        Args:
            x: Input RGB image tensor [B, 3, H, W]
            
        Returns:
            NSST feature tensor [B, 87, H, W]
            - 87 = 29 channels per RGB channel
            - 29 = 1 final LP + (2 + 4 + 8) directional bands from 3 scales
        """
        batch_size, in_channels, H, W = x.shape
        assert in_channels == 3, f"Expected 3 input channels (RGB), got {in_channels}"
        
        all_features = []
        
        # Process each color channel independently
        for c in range(in_channels):
            channel_input = x[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # Step 1: Non-Subsampled Laplacian Pyramid
            pyramid = self._nonsubsampled_laplacian_pyramid(channel_input)
            # pyramid = [bandpass_0, bandpass_1, bandpass_2, lowpass_final]
            
            channel_features = []
            
            # Step 2: Directional decomposition for each bandpass
            for scale_idx in range(self.scales):
                bandpass = pyramid[scale_idx]
                # Decompose bandpass into directional subbands
                directional_subbands = self._directional_decomposition(bandpass, scale_idx)
                channel_features.extend(directional_subbands)
            
            # Step 3: Add final lowpass residual
            channel_features.append(pyramid[-1])  # Final lowpass
            
            # Concatenate all features for this channel: [B, 29, H, W]
            channel_stack = torch.cat(channel_features, dim=1)
            all_features.append(channel_stack)
        
        # Concatenate features from all RGB channels: [B, 87, H, W]
        nsst_features = torch.cat(all_features, dim=1)
        
        return nsst_features


def nsct_decomposition_to_tensor(rgb_batch: torch.Tensor, 
                                  scales: int = 3,
                                  directions_profile: List[int] = None) -> torch.Tensor:
    """
    Convenience function for NSST feature extraction.
    
    Args:
        rgb_batch: Input RGB batch [N, 3, H, W]
        scales: Number of decomposition scales
        directions_profile: Directional decomposition profile [2, 3, 4] for 87 channels
        
    Returns:
        NSST features [N, 87, H, W]
    """
    if directions_profile is None:
        directions_profile = [2, 3, 4]  # Default gives 87 channels: (4+8+16+1)*3 = 87
    
    device = rgb_batch.device
    nsst = NSSTDecomposition(scales=scales, directions_profile=directions_profile).to(device)
    
    with torch.no_grad():
        nsst.eval()
        features = nsst(rgb_batch)
    
    return features


# Utility function for verification
def verify_nsst_output_shape(input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512)):
    """
    Verify NSST output shape matches expected dimensions.
    
    Args:
        input_shape: Input tensor shape (B, C, H, W)
    """
    x = torch.randn(*input_shape)
    nsst = NSSTDecomposition(scales=3, directions_profile=[2, 3, 4])
    
    output = nsst(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({input_shape[0]}, 87, {input_shape[2]}, {input_shape[3]})")
    print(f"Channels per input: {nsst.channels_per_input}")
    print(f"Total output channels: {nsst.channels_per_input * 3}")
    
    assert output.shape == (input_shape[0], 87, input_shape[2], input_shape[3]), \
        f"Shape mismatch! Got {output.shape}"
    
    print("✓ NSST output shape verification passed!")
    
    return output


if __name__ == "__main__":
    # Test the NSST decomposition
    verify_nsst_output_shape()
