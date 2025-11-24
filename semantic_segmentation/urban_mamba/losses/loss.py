"""
Loss functions for UrbanMamba v3 training.
Combines Cross-Entropy and Lovász-Softmax losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CompositeLoss(nn.Module):
    """
    Composite loss combining Cross-Entropy and Lovász-Softmax.
    
    Args:
        num_classes: Number of segmentation classes
        ce_weight: Weight for cross-entropy loss (default: 0.7)
        lovasz_weight: Weight for Lovász loss (default: 0.3)
        ignore_index: Index to ignore in loss computation (default: -100)
    """
    
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.7,
        lovasz_weight: float = 0.3,
        ignore_index: int = 255
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index
        
        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def lovasz_softmax(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Lovász-Softmax loss (multi-class).
        
        Args:
            probas: Predicted probabilities [B, C, H, W]
            labels: Ground truth labels [B, H, W]
        
        Returns:
            Loss value (scalar)
        """
        B, C, H, W = probas.shape
        
        # Flatten spatial dimensions
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = labels.reshape(-1)  # [B*H*W]
        
        # Filter out ignore_index
        if self.ignore_index is not None:
            valid_mask = labels_flat != self.ignore_index
            probas_flat = probas_flat[valid_mask]
            labels_flat = labels_flat[valid_mask]
        
        if probas_flat.numel() == 0:
            return torch.tensor(0.0, device=probas.device)
        
        # Compute Lovász loss per class and average
        losses = []
        for c in range(C):
            # Binary mask for class c
            fg = (labels_flat == c).float()
            
            if fg.sum() == 0:
                continue
            
            # Class probability errors
            errors = (1 - probas_flat[:, c]) * fg + probas_flat[:, c] * (1 - fg)
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            
            # Lovász extension
            grad = self._lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probas.device)
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of the Lovász extension.
        
        Args:
            gt_sorted: Sorted ground truth (0 or 1) [N]
        
        Returns:
            Gradient values [N]
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1 - intersection / union
        
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
        return jaccard
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Dictionary with 'loss', 'loss_ce', 'loss_lovasz'
        """
        # Cross-entropy loss
        loss_ce = self.ce_loss(pred, target)
        
        # Lovász-Softmax loss
        probas = F.softmax(pred, dim=1)
        loss_lovasz = self.lovasz_softmax(probas, target)
        
        # Composite loss
        loss = self.ce_weight * loss_ce + self.lovasz_weight * loss_lovasz
        
        return {
            'loss': loss,
            'loss_ce': loss_ce,
            'loss_lovasz': loss_lovasz
        }


if __name__ == "__main__":
    # Test CompositeLoss
    print("Testing CompositeLoss...")
    
    criterion = CompositeLoss(num_classes=6)
    
    # Create dummy data
    B, C, H, W = 2, 6, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Compute loss
    loss_dict = criterion(pred, target)
    
    print(f"\nLoss values:")
    print(f"  Total: {loss_dict['loss']:.4f}")
    print(f"  CE: {loss_dict['loss_ce']:.4f}")
    print(f"  Lovász: {loss_dict['loss_lovasz']:.4f}")
    
    # Test backward
    loss_dict['loss'].backward()
    print(f"\n✓ Backward pass successful!")
