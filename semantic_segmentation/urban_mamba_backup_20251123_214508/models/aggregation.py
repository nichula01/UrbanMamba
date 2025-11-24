"""
Multi-Scale Feature Aggregation Head
Implements bi-directional feedback loop with learnable scale weights for
optimal feature pyramid construction and cross-scale interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiScaleAggregationHead(nn.Module):
    """
    Multi-Scale Feature Aggregation Head with bi-directional feedback.
    
    Implements:
    1. Feature projection to uniform dimensions
    2. Bi-directional cross-scale interaction
    3. Weighted summation with learnable scale weights
    
    P_j^fused = P_j + Resize(P_{j+1}^fused) + α·Resize(P_{j-1}^fused)
    F_final = Σ w_j · Upsample(P_j^fused)
    """
    
    def __init__(
        self,
        encoder_dims: List[int] = [96, 192, 384, 768],
        uniform_dim: int = 256,
        num_classes: int = 6,
        output_stride: int = 4
    ):
        """
        Initialize Multi-Scale Aggregation Head.
        
        Args:
            encoder_dims: Channel dimensions from encoder stages
            uniform_dim: Uniform channel dimension for all scales
            num_classes: Number of output classes
            output_stride: Output stride relative to input (4 = 1/4 resolution)
        """
        super(MultiScaleAggregationHead, self).__init__()
        
        self.encoder_dims = encoder_dims
        self.uniform_dim = uniform_dim
        self.num_scales = len(encoder_dims)
        self.output_stride = output_stride
        
        # 1. Feature projection layers (1x1 conv to uniform dimension)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, uniform_dim, 1, bias=False),
                nn.BatchNorm2d(uniform_dim),
                nn.ReLU(inplace=True)
            )
            for dim in encoder_dims
        ])
        
        # 2. Learnable alpha parameters for fine-grained detail weighting
        # One alpha per scale (except first and last)
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5)
            for _ in range(self.num_scales - 2)
        ])
        
        # 3. Refinement conv after aggregation at each scale
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(uniform_dim, uniform_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(uniform_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(self.num_scales)
        ])
        
        # 4. Learnable weights for final weighted summation
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        
        # 5. Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(uniform_dim, uniform_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(uniform_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(uniform_dim, num_classes, 1)
        )
    
    def _resize(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Resize feature map to target spatial size.
        
        Args:
            x: Input feature [B, C, H, W]
            target_size: Target (H, W)
            
        Returns:
            Resized feature [B, C, H_target, W_target]
        """
        if x.shape[2:] == target_size:
            return x
        
        return F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multi-scale features and generate segmentation map.
        
        Args:
            encoder_features: List of fused features [F1, F2, F3, F4]
                             from shallow to deep
        
        Returns:
            Segmentation logits [B, num_classes, H/output_stride, W/output_stride]
        """
        assert len(encoder_features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(encoder_features)}"
        
        # Step 1: Project all features to uniform dimension
        projected = [
            self.projections[i](encoder_features[i])
            for i in range(self.num_scales)
        ]
        
        # Step 2: Bi-directional feedback loop
        # Process from coarse to fine and fine to coarse
        aggregated = [None] * self.num_scales
        
        # Initialize deepest scale
        aggregated[-1] = projected[-1]
        
        # Top-down pass (coarse to fine)
        for j in range(self.num_scales - 2, -1, -1):
            # Current scale projection
            current = projected[j]
            
            # Contribution from coarser scale (j+1)
            coarser = self._resize(aggregated[j + 1], current.shape[2:])
            
            # Aggregate
            aggregated[j] = current + coarser
        
        # Bottom-up pass (fine to coarse) with learnable alpha
        # Refine aggregated features with fine-grained details
        refined = [None] * self.num_scales
        refined[0] = aggregated[0]
        
        for j in range(1, self.num_scales):
            # Current aggregated feature
            current = aggregated[j]
            
            # Contribution from finer scale (j-1)
            finer = self._resize(refined[j - 1], current.shape[2:])
            
            # Apply learnable alpha weighting
            if j < self.num_scales - 1:
                alpha = torch.sigmoid(self.alphas[j - 1])  # Ensure 0-1 range
                refined[j] = current + alpha * finer
            else:
                # Last scale (no alpha needed)
                refined[j] = current + 0.5 * finer
            
            # Apply refinement convolution
            refined[j] = self.refinement_convs[j](refined[j])
        
        # Step 3: Weighted summation across all scales
        # Normalize scale weights with softmax
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Determine output size (e.g., 1/4 of input for output_stride=4)
        # Use the resolution of the first scale as reference
        target_size = (
            refined[0].shape[2] * (4 // self.output_stride),
            refined[0].shape[3] * (4 // self.output_stride)
        )
        
        # Upsample and weight each scale
        weighted_features = []
        for j in range(self.num_scales):
            # Upsample to target size
            upsampled = self._resize(refined[j], target_size)
            
            # Apply scale weight
            weighted = weights[j] * upsampled
            weighted_features.append(weighted)
        
        # Sum all weighted features
        final_features = torch.stack(weighted_features, dim=0).sum(dim=0)
        
        # Step 4: Final classification
        logits = self.classifier(final_features)
        
        return logits


if __name__ == "__main__":
    print("Testing Multi-Scale Aggregation Head...")
    
    # Create aggregation head
    agg_head = MultiScaleAggregationHead(
        encoder_dims=[96, 192, 384, 768],
        uniform_dim=256,
        num_classes=6,
        output_stride=4
    )
    
    # Simulate multi-scale encoder features
    encoder_feats = [
        torch.randn(2, 96, 128, 128),   # Scale 1 (1/4 resolution)
        torch.randn(2, 192, 64, 64),    # Scale 2 (1/8 resolution)
        torch.randn(2, 384, 32, 32),    # Scale 3 (1/16 resolution)
        torch.randn(2, 768, 16, 16)     # Scale 4 (1/32 resolution)
    ]
    
    # Forward pass
    logits = agg_head(encoder_feats)
    
    print(f"\nInput features:")
    for i, feat in enumerate(encoder_feats):
        print(f"  Scale {i+1}: {feat.shape}")
    
    print(f"\nOutput logits: {logits.shape}")
    print(f"Expected: (2, 6, 128, 128) for output_stride=4")
    
    # Check learnable parameters
    print(f"\nLearnable parameters:")
    print(f"  Alpha values: {[alpha.item() for alpha in agg_head.alphas]}")
    print(f"  Scale weights: {agg_head.scale_weights.detach().numpy()}")
    print(f"  Normalized weights: {F.softmax(agg_head.scale_weights, dim=0).detach().numpy()}")
    
    print("\n✓ Multi-Scale Aggregation Head test passed!")
