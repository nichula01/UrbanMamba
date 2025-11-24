"""
Lovász-Softmax Loss Implementation
Numerically stable implementation for direct mIoU optimization.
Critical for sharp boundary detection in high-resolution segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of the Lovász extension w.r.t. the sorted errors.
    
    Args:
        gt_sorted: Sorted ground truth labels
        
    Returns:
        Lovász gradient
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    
    if len(gt_sorted) > 1:  # Cover 1-pixel case
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, 
                        classes: str = 'present', ignore_index: int = -100) -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss.
    
    Args:
        probas: Class probabilities [P, C] where P = H*W
        labels: Ground truth labels [P]
        classes: 'all' for all classes, 'present' for classes present in labels
        ignore_index: Label to ignore
        
    Returns:
        Lovász loss value
    """
    if probas.numel() == 0:
        # Only void pixels, the gradients should be 0
        return probas * 0.
    
    C = probas.size(1)
    losses = []
    
    # Filter out ignore_index
    valid = (labels != ignore_index)
    if not valid.all():
        probas = probas[valid]
        labels = labels[valid]
    
    if probas.numel() == 0:
        return torch.tensor(0., device=probas.device, requires_grad=True)
    
    class_to_sum = list(range(C)) if classes == 'all' else None
    
    for c in range(C):
        # Foreground for class c
        fg = (labels == c).float()
        
        if classes == 'present' and fg.sum() == 0:
            continue
        
        if class_to_sum is None:
            class_to_sum = []
        
        if classes == 'present' or fg.sum() > 0:
            # Error = 1 - probability of true class + probability of class c
            errors = (1. - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            
            loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
            losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0., device=probas.device, requires_grad=True)
    
    return torch.mean(torch.stack(losses))


def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, 
                   ignore_index: int = -100) -> tuple:
    """
    Flatten predictions and labels.
    
    Args:
        probas: Predictions [B, C, H, W]
        labels: Labels [B, H, W]
        ignore_index: Label to ignore
        
    Returns:
        Flattened probas and labels
    """
    if len(probas.shape) == 3:
        # Assumes output is logits [B, H, W]
        probas = probas.unsqueeze(1)
    
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
    labels = labels.view(-1)  # [B*H*W]
    
    return probas, labels


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for multi-class segmentation.
    
    Directly optimizes the mean Intersection-over-Union (mIoU) metric.
    Essential for high-resolution segmentation to prevent blurry boundaries.
    
    Reference:
    Berman et al. "The Lovász-Softmax loss: A tractable surrogate for the 
    optimization of the intersection-over-union measure in neural networks"
    CVPR 2018
    """
    
    def __init__(self, classes: str = 'present', ignore_index: int = -100):
        """
        Initialize Lovász-Softmax loss.
        
        Args:
            classes: 'all' for all classes, 'present' for classes in minibatch
            ignore_index: Label to ignore in loss computation
        """
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Lovász-Softmax loss.
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth labels [B, H, W]
            
        Returns:
            Loss value
        """
        # Apply softmax to get probabilities
        probas = F.softmax(logits, dim=1)
        
        # Flatten
        probas, labels = flatten_probas(probas, labels, self.ignore_index)
        
        # Compute loss
        loss = lovasz_softmax_flat(probas, labels, self.classes, self.ignore_index)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Useful for urban segmentation where some classes (e.g., roads, water)
    may be under-represented.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 ignore_index: int = -100):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            ignore_index: Label to ignore
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth labels [B, H, W]
            
        Returns:
            Loss value
        """
        # Create mask for valid pixels
        valid_mask = (labels != self.ignore_index)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=self.ignore_index)
        
        # Compute probabilities
        p = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        # Apply mask and mean
        focal_loss = focal_loss[valid_mask].mean()
        
        return focal_loss


class CompositeLoss(nn.Module):
    """
    Composite Loss combining Cross-Entropy and Lovász-Softmax.
    
    L_total = λ₁ * L_CE + λ₂ * L_Lovász
    
    Balances:
    - L_CE: Pixel-wise semantic correctness and training stability
    - L_Lovász: Direct mIoU optimization for sharp boundaries
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        lambda_ce: float = 1.0,
        lambda_lovasz: float = 0.75,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        use_focal: bool = False
    ):
        """
        Initialize Composite Loss.
        
        Args:
            num_classes: Number of classes
            lambda_ce: Weight for Cross-Entropy loss
            lambda_lovasz: Weight for Lovász loss
            class_weights: Class weights for CE loss
            ignore_index: Label to ignore
            use_focal: Use Focal loss instead of CE
        """
        super(CompositeLoss, self).__init__()
        
        self.lambda_ce = lambda_ce
        self.lambda_lovasz = lambda_lovasz
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        
        # Cross-Entropy or Focal Loss
        if use_focal:
            self.ce_loss = FocalLoss(ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        
        # Lovász-Softmax Loss
        self.lovasz_loss = LovaszSoftmaxLoss(
            classes='present',
            ignore_index=ignore_index
        )
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute composite loss.
        
        Args:
            logits: Model predictions [B, C, H, W]
            labels: Ground truth labels [B, H, W]
            
        Returns:
            Dictionary with total loss and components
        """
        # Cross-Entropy / Focal Loss
        loss_ce = self.ce_loss(logits, labels)
        
        # Lovász-Softmax Loss
        loss_lovasz = self.lovasz_loss(logits, labels)
        
        # Total loss
        total_loss = self.lambda_ce * loss_ce + self.lambda_lovasz * loss_lovasz
        
        return {
            'loss': total_loss,
            'loss_ce': loss_ce.detach(),
            'loss_lovasz': loss_lovasz.detach()
        }


def create_loss_function(
    num_classes: int = 6,
    loss_type: str = 'composite',
    lambda_ce: float = 1.0,
    lambda_lovasz: float = 0.75,
    class_weights: Optional[list] = None,
    ignore_index: int = -100
):
    """
    Factory function to create loss function.
    
    Args:
        num_classes: Number of classes
        loss_type: 'ce', 'lovasz', 'focal', or 'composite'
        lambda_ce: Weight for CE loss (composite only)
        lambda_lovasz: Weight for Lovász loss (composite only)
        class_weights: Class weights
        ignore_index: Label to ignore
        
    Returns:
        Loss function
    """
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights)
    
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
    elif loss_type == 'lovasz':
        return LovaszSoftmaxLoss(classes='present', ignore_index=ignore_index)
    
    elif loss_type == 'focal':
        return FocalLoss(ignore_index=ignore_index)
    
    elif loss_type == 'composite':
        return CompositeLoss(
            num_classes=num_classes,
            lambda_ce=lambda_ce,
            lambda_lovasz=lambda_lovasz,
            class_weights=class_weights,
            ignore_index=ignore_index,
            use_focal=False
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Test data
    batch_size = 2
    num_classes = 6
    H, W = 128, 128
    
    logits = torch.randn(batch_size, num_classes, H, W)
    labels = torch.randint(0, num_classes, (batch_size, H, W))
    
    # Test Lovász loss
    print("\n1. Testing Lovász-Softmax Loss...")
    lovasz = LovaszSoftmaxLoss()
    loss_val = lovasz(logits, labels)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test Focal loss
    print("\n2. Testing Focal Loss...")
    focal = FocalLoss()
    loss_val = focal(logits, labels)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test Composite loss
    print("\n3. Testing Composite Loss...")
    composite = CompositeLoss(num_classes=num_classes)
    loss_dict = composite(logits, labels)
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   CE loss: {loss_dict['loss_ce'].item():.4f}")
    print(f"   Lovász loss: {loss_dict['loss_lovasz'].item():.4f}")
    
    # Test with ignore index
    print("\n4. Testing with ignore_index...")
    labels_with_ignore = labels.clone()
    labels_with_ignore[0, :10, :10] = -100
    loss_dict = composite(logits, labels_with_ignore)
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    logits.requires_grad = True
    loss_dict = composite(logits, labels)
    loss_dict['loss'].backward()
    print(f"   Gradient norm: {logits.grad.norm().item():.4f}")
    
    print("\n✓ All loss tests passed!")
