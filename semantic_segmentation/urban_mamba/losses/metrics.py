"""
Evaluation metrics for semantic segmentation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class SegmentationMetrics:
    """
    Compute segmentation metrics (mIoU, pixel accuracy, class accuracy).
    
    Args:
        num_classes: Number of segmentation classes
        ignore_index: Index to ignore in metric computation (default: -100)
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted logits [B, C, H, W] or labels [B, H, W]
            target: Ground truth labels [B, H, W]
        """
        # Convert logits to labels if needed
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        # Flatten
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Filter out ignore_index
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            pred = pred[valid_mask]
            target = target[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with 'mIoU', 'pixel_acc', 'mean_acc', 'per_class_iou'
        """
        # Per-class IoU
        iou_per_class = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if (tp + fp + fn) > 0:
                iou_per_class[i] = tp / (tp + fp + fn)
            else:
                iou_per_class[i] = 0.0
        
        # Mean IoU
        miou = np.mean(iou_per_class)
        
        # Pixel accuracy
        pixel_acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        
        # Mean class accuracy
        class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        mean_acc = np.mean(class_acc)
        
        return {
            'mIoU': float(miou),
            'pixel_acc': float(pixel_acc),
            'mean_acc': float(mean_acc),
            'per_class_iou': iou_per_class.tolist()
        }


if __name__ == "__main__":
    # Test SegmentationMetrics
    print("Testing SegmentationMetrics...")
    
    metrics = SegmentationMetrics(num_classes=6)
    
    # Create dummy data
    B, C, H, W = 2, 6, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Update metrics
    metrics.update(pred, target)
    
    # Get results
    results = metrics.get_metrics()
    
    print(f"\nMetrics:")
    print(f"  mIoU: {results['mIoU']:.4f}")
    print(f"  Pixel Acc: {results['pixel_acc']:.4f}")
    print(f"  Mean Acc: {results['mean_acc']:.4f}")
    print(f"  Per-class IoU: {[f'{x:.4f}' for x in results['per_class_iou']]}")
    
    print(f"\n✓ Metrics test passed!")
