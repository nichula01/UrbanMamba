import torch
import torch.nn as nn
import torch.nn.functional as F

from .nsst_utils import NSSTDecomposer


class NSSTFreqEncoderCNN(nn.Module):
    """
    NSST-based frequency encoder using a compact CNN.

    - RGB -> luminance -> NSST (3 scales, [4,4,8] dirs).
    - All bands (1 LF + 4 + 4 + 8 HF) are stacked as 17 channels.
    - A 4-stage CNN encoder produces multi-scale feature maps for fusion
      with the RGB-VMamba encoder.
    """

    def __init__(
        self,
        directions_per_scale=(4, 4, 8),
        in_channels=17,
        out_channels_list=(64, 128, 256, 512),
    ):
        super().__init__()
        self.decomposer = NSSTDecomposer(directions_per_scale=directions_per_scale)
        self.out_channels_list = out_channels_list

        c1, c2, c3, c4 = out_channels_list

        # Stage 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2

        # Stage 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4

        # Stage 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/8, W/8

        # Stage 4 (deepest)
        self.enc4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

    def _rgb_to_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        R = rgb[:, 0:1, :, :]
        G = rgb[:, 1:2, :, :]
        B = rgb[:, 2:3, :, :]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        return Y

    def _nsst_stack(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W)
        Returns:
            x: (B, 17, H, W) stacked NSST bands.
        """
        B, _, H, W = rgb.shape
        Y = self._rgb_to_luminance(rgb)  # (B,1,H,W)

        bands_list = []
        for b in range(B):
            y = Y[b]  # (1, H, W)
            lf, hfs = self.decomposer(y)
            hf_s1, hf_s2, hf_s3 = hfs  # (4,H,W), (4,H,W), (8,H,W)

            x_b = torch.cat([lf, hf_s1, hf_s2, hf_s3], dim=0)  # (17,H,W)
            bands_list.append(x_b)

        x = torch.stack(bands_list, dim=0)  # (B,17,H,W)
        return x

    def forward(self, rgb: torch.Tensor):
        """
        Args:
            rgb: (B, 3, H, W) input RGB image (same as main encoder input).
        Returns:
            feats: list [f1, f2, f3, f4] of multi-scale features:
                f1: (B, C1, H,   W)
                f2: (B, C2, H/2, W/2)
                f3: (B, C3, H/4, W/4)
                f4: (B, C4, H/8, W/8)
        """
        x = self._nsst_stack(rgb)  # (B,17,H,W)

        f1 = self.enc1(x)
        x = self.down1(f1)
        f2 = self.enc2(x)
        x = self.down2(f2)
        f3 = self.enc3(x)
        x = self.down3(f3)
        f4 = self.enc4(x)

        return [f1, f2, f3, f4]
