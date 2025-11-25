"""
Loss functions for UrbanMamba v3 training.
Combines cross-entropy, Lovász-Softmax, and boundary-aware losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BoundaryDiceLoss(nn.Module):
    """
    Multi-class boundary-aware Dice loss.

    It extracts soft boundary maps from both prediction and target, and
    computes a Dice loss over these boundaries. This encourages sharp,
    accurate edges, which is important for urban structures such as roads
    and buildings.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    def _compute_soft_boundary(
        self,
        prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prob: [B, C, H, W] softmax probabilities.
        Returns:
            boundary map [B, C, H, W] with larger values near class boundaries.
        """

        kernel = self.laplacian_kernel.expand(self.num_classes, 1, 3, 3)
        boundary = F.conv2d(prob, kernel, padding=1, groups=self.num_classes)
        return boundary.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(pred, dim=1)
        B, C, H, W = prob.shape
        valid_mask = target != self.ignore_index
        safe_target = target.clone()
        safe_target = safe_target.clamp(0, C - 1)
        one_hot = F.one_hot(safe_target, num_classes=C).permute(0, 3, 1, 2).to(prob.dtype)
        one_hot = one_hot * valid_mask.unsqueeze(1)

        pred_bnd = self._compute_soft_boundary(prob)
        target_bnd = self._compute_soft_boundary(one_hot)

        pred_flat = pred_bnd.view(B, C, -1)
        target_flat = target_bnd.view(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=-1)
        denom = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) + self.eps
        dice = (2.0 * intersection + self.eps) / denom

        return 1.0 - dice.mean()


class CompositeLoss(nn.Module):
    """
    Composite loss for urban remote sensing segmentation.

    It blends class-balanced cross-entropy (with optional label smoothing),
    Lovász-Softmax IoU, and a boundary-aware Dice term to cope with class
    imbalance and to promote crisp building/road edges.
    """
    
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.6,
        lovasz_weight: float = 0.3,
        boundary_weight: float = 0.1,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        use_dynamic_class_weights: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.use_dynamic_class_weights = use_dynamic_class_weights
        self.label_smoothing = label_smoothing

        self.boundary_loss_fn = BoundaryDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    
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
    
    def _compute_dynamic_class_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-class weights from current batch labels (ignore void pixels).

        Args:
            target: Ground truth labels [B, H, W].

        Returns:
            Tensor of shape [num_classes] with inverse-log frequency weights.
        """
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.ones(self.num_classes, device=target.device)

        valid_labels = target[valid_mask].view(-1)
        counts = torch.bincount(valid_labels, minlength=self.num_classes)
        freq = counts.float() / (counts.sum() + 1e-6)
        weights = 1.0 / torch.log(1.02 + freq)
        return weights.to(target.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(pred.device)
        elif self.use_dynamic_class_weights:
            weight = self._compute_dynamic_class_weights(target)

        loss_ce = F.cross_entropy(
            pred,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        probas = F.softmax(pred, dim=1)
        loss_lovasz = self.lovasz_softmax(probas, target)
        loss_boundary = self.boundary_loss_fn(pred, target)

        loss = (
            self.ce_weight * loss_ce
            + self.lovasz_weight * loss_lovasz
            + self.boundary_weight * loss_boundary
        )
        
        return {
            'loss': loss,
            'loss_ce': loss_ce,
            'loss_lovasz': loss_lovasz,
            'loss_boundary': loss_boundary
        }


if __name__ == "__main__":
    print("Testing CompositeLoss...")

    B, C, H, W = 2, 6, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    criterion = CompositeLoss(
        num_classes=C,
        ce_weight=0.6,
        lovasz_weight=0.3,
        boundary_weight=0.1,
        use_dynamic_class_weights=True,
        label_smoothing=0.1,
    )

    loss_dict = criterion(pred, target)

    print("Loss values:")
    print(f"  Total: {loss_dict['loss']:.4f}")
    print(f"  CE: {loss_dict['loss_ce']:.4f}")
    print(f"  Lovász: {loss_dict['loss_lovasz']:.4f}")
    print(f"  Boundary: {loss_dict['loss_boundary']:.4f}")

    loss_dict['loss'].backward()
    print("\n✓ Backward pass successful!")
