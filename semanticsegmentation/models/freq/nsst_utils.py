import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def _make_direction_kernels(num_dirs: int, device: torch.device, dtype: torch.dtype):
    """Create simple oriented gradient kernels by rotating Sobel filters."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)
    angles = torch.linspace(0, math.pi, steps=num_dirs + 1, device=device, dtype=dtype)[:-1]
    kernels = []
    for theta in angles:
        k = torch.cos(theta) * kx + torch.sin(theta) * ky
        k = k / (k.abs().sum() + 1e-6)
        kernels.append(k)
    return torch.stack(kernels, dim=0)  # (num_dirs, 3, 3)


def _blur_single(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Box blur preserving spatial size."""
    pad = ksize // 2
    kernel = torch.ones((1, 1, ksize, ksize), device=x.device, dtype=x.dtype) / float(ksize * ksize)
    return F.conv2d(x.unsqueeze(0), kernel, padding=pad).squeeze(0)


class NSSTDecomposer(object):
    """
    Lightweight NSST-style decomposition on a single-channel image.
    Uses 3 scales with [4, 4, 8] directions (fine -> coarse).

    For an input (1, H, W) tensor x (grayscale),
    returns:
        lf: (1, H, W)  low-frequency band
        hfs: list [hf_s1, hf_s2, hf_s3]
             hf_s1: (4, H, W)
             hf_s2: (4, H, W)
             hf_s3: (8, H, W)
    """

    def __init__(self, directions_per_scale=(4, 4, 8)):
        self.directions_per_scale = directions_per_scale

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (1, H, W) float tensor, grayscale image.
        Returns:
            lf: (1, H, W)
            hfs: [hf_s1, hf_s2, hf_s3]
        """
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected grayscale tensor of shape (1, H, W), got {x.shape}")

        device, dtype = x.device, x.dtype
        # Low-frequency via modest blur
        lf = _blur_single(x, ksize=7)

        hfs = []
        # Use increasingly smoothed inputs for coarser scales
        scale_blurs = [3, 5, 7]
        for dirs, blur_k in zip(self.directions_per_scale, scale_blurs):
            x_blur = _blur_single(x, ksize=blur_k)
            kernels = _make_direction_kernels(dirs, device, dtype).unsqueeze(1)  # (D,1,3,3)
            resp = F.conv2d(x_blur.unsqueeze(0), kernels, padding=1).squeeze(0)  # (D,H,W)
            hfs.append(resp)

        return lf, hfs


def nsst_four_bands_from_rgb(rgb_tensor: torch.Tensor, decomposer: NSSTDecomposer):
    """
    Convert an RGB tensor to luminance and apply NSST, then aggregate
    into 4 effective bands: [LL, B1, B2, B3].

    Args:
        rgb_tensor: (B, 3, H, W) float tensor.
        decomposer: NSSTDecomposer instance.

    Returns:
        bands: list of 4 tensors [band0, band1, band2, band3],
               each of shape (B, 1, H, W).
    """
    if rgb_tensor.dim() != 4 or rgb_tensor.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor of shape (B,3,H,W), got {rgb_tensor.shape}")

    B, _, H, W = rgb_tensor.shape
    R = rgb_tensor[:, 0:1, :, :]
    G = rgb_tensor[:, 1:2, :, :]
    Bc = rgb_tensor[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * Bc  # (B,1,H,W)

    lf_list = []
    b1_list = []
    b2_list = []
    b3_list = []

    for b in range(B):
        y = Y[b]  # (1, H, W)
        lf, hfs = decomposer(y)
        hf_s1, hf_s2, hf_s3 = hfs

        b1 = hf_s1.mean(dim=0, keepdim=True)
        b2 = hf_s2.mean(dim=0, keepdim=True)
        b3 = hf_s3.mean(dim=0, keepdim=True)

        lf_list.append(lf)
        b1_list.append(b1)
        b2_list.append(b2)
        b3_list.append(b3)

    band0 = torch.stack(lf_list, dim=0)  # (B,1,H,W)
    band1 = torch.stack(b1_list, dim=0)  # (B,1,H,W)
    band2 = torch.stack(b2_list, dim=0)  # (B,1,H,W)
    band3 = torch.stack(b3_list, dim=0)  # (B,1,H,W)

    return [band0, band1, band2, band3]
