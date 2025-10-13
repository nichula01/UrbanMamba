"""
Utilities for performing basic wavelet operations.

This module implements a simple 2‑D Haar discrete wavelet transform (DWT)
for use in the UrbanMamba architecture.  The transform splits an input
feature map into four sub‑bands: a low–low approximation (LL) and three
detail components (LH, HL and HH)【434926937949839†L121-L133】.  Each band
has half the height and width of the input.  The decomposition is
implemented as described in the UrbanMamba methodology and does not
require any external dependencies beyond PyTorch.

Usage:

```
from utils_wavelet import haar_wavelet_decompose

# x is a tensor of shape (B, C, H, W) with H and W divisible by 2
ll, lh, hl, hh = haar_wavelet_decompose(x)
```

All operations are differentiable and support back‑propagation.
"""

import torch

def haar_wavelet_decompose(x: torch.Tensor):
    """Perform a single‑level 2‑D Haar discrete wavelet decomposition.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).  The height
            and width must be even so that the signal can be split into
            2×2 blocks【434926937949839†L121-L133】.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The LL, LH, HL and HH sub‑bands.  Each has shape
            (B, C, H/2, W/2).  The decomposition is implemented using
            strided slicing and additions/subtractions.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected input of shape (B, C, H, W), got {x.shape}")
    _, _, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(
            "The height and width of the input must be divisible by 2 for the Haar DWT."
        )
    # Sample even and odd rows/columns
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    # Compute approximation and detail coefficients
    ll = 0.25 * (x00 + x01 + x10 + x11)
    lh = 0.25 * (x00 + x01 - x10 - x11)
    hl = 0.25 * (x00 - x01 + x10 - x11)
    hh = 0.25 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh