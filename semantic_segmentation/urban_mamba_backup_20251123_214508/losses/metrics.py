"""
Evaluation Metrics for Urban Semantic Segmentation
Implements mIoU, per-class IoU, and boundary F1-score.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from scipy.ndimage import distance_transform_edt, binary_erosion


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -100, 
                 class_names: Optional[List[str]] = None):
        """
        Initialize metrics.
        
        Args:
            num_classes: Number of classes
            ignore_index: Label to ignore
            class_names: Names of classes for reporting
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W]
            labels: Ground truth [B, H, W]
        """
        # Convert logits to class predictions
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Filter out ignore_index
        mask = (labels != self.ignore_index)
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Update confusion matrix
        for pred_class in range(self.num_classes):
            for true_class in range(self.num_classes):
                self.confusion_matrix[true_class, pred_class] += \
                    np.sum((predictions == pred_class) & (labels == true_class))
    
    def update_boundary(self, predictions: torch.Tensor, labels: torch.Tensor, 
                       threshold: int = 2):
        """
        Update boundary F1 metrics.
        
        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W]
            labels: Ground truth [B, H, W]
            threshold: Distance threshold for boundary matching (pixels)
        """
        # Convert logits to class predictions
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)
        
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        batch_size = predictions.shape[0]
        
        for b in range(batch_size):
            pred = predictions[b]
            label = labels[b]
            
            # Extract boundaries
            pred_boundary = self._extract_boundaries(pred)
            label_boundary = self._extract_boundaries(label)
            
            if label_boundary.sum() == 0:
                continue
            
            # Compute distance transforms
            pred_dist = distance_transform_edt(~pred_boundary)
            label_dist = distance_transform_edt(~label_boundary)
            
            # True positives: predicted boundaries within threshold of true boundaries
            tp = np.sum((pred_boundary > 0) & (label_dist <= threshold))
            
            # False positives: predicted boundaries far from true boundaries
            fp = np.sum((pred_boundary > 0) & (label_dist > threshold))
            
            # False negatives: true boundaries far from predicted boundaries
            fn = np.sum((label_boundary > 0) & (pred_dist > threshold))
            
            self.boundary_tp += tp
            self.boundary_fp += fp
            self.boundary_fn += fn
    
    def _extract_boundaries(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Extract boundaries from segmentation map using morphological operations.
        
        Args:
            segmentation: Segmentation map [H, W]
            
        Returns:
            Binary boundary map [H, W]
        """
        # Create binary boundary map
        boundary = np.zeros_like(segmentation, dtype=bool)
        
        # For each class, find boundaries
        for c in range(self.num_classes):
            mask = (segmentation == c)
            if mask.sum() > 0:
                eroded = binary_erosion(mask)
                class_boundary = mask & ~eroded
                boundary |= class_boundary
        
        return boundary
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute IoU metrics.
        
        Returns:
            Dictionary with mIoU and per-class IoU
        """
        # Intersection = diagonal of confusion matrix
        intersection = np.diag(self.confusion_matrix)
        
        # Union = sum of row + sum of column - intersection
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - 
                intersection)
        
        # IoU per class
        iou_per_class = intersection / (union + 1e-10)
        
        # Mean IoU (only for classes present in dataset)
        valid_classes = union > 0
        mean_iou = iou_per_class[valid_classes].mean()
        
        # Create results dictionary
        results = {'mIoU': mean_iou}
        
        for i, class_name in enumerate(self.class_names):
            results[f'IoU_{class_name}'] = iou_per_class[i]
        
        return results
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute overall pixel accuracy.
        
        Returns:
            Pixel accuracy
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        return correct / (total + 1e-10)
    
    def compute_boundary_f1(self) -> Dict[str, float]:
        """
        Compute boundary F1-score.
        
        Returns:
            Dictionary with precision, recall, and F1
        """
        precision = self.boundary_tp / (self.boundary_tp + self.boundary_fp + 1e-10)
        recall = self.boundary_tp / (self.boundary_tp + self.boundary_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1
        }
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # IoU metrics
        iou_metrics = self.compute_iou()
        results.update(iou_metrics)
        
        # Pixel accuracy
        results['pixel_accuracy'] = self.compute_pixel_accuracy()
        
        # Boundary F1
        if self.boundary_tp + self.boundary_fp + self.boundary_fn > 0:
            boundary_metrics = self.compute_boundary_f1()
            results.update(boundary_metrics)
        
        return results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_matrix.copy()


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                   num_classes: int, ignore_index: int = -100) -> Dict[str, float]:
    """
    Compute metrics for a single batch (utility function).
    
    Args:
        predictions: Model predictions [B, C, H, W]
        labels: Ground truth [B, H, W]
        num_classes: Number of classes
        ignore_index: Label to ignore
        
    Returns:
        Dictionary with metrics
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(predictions, labels)
    return metrics.compute_all()


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Test data
    num_classes = 6
    class_names = ['buildings', 'roads', 'vegetation', 'water', 'bare_land', 'other']
    
    # Create metrics object
    metrics = SegmentationMetrics(num_classes, class_names=class_names)
    
    # Simulate predictions and labels
    batch_size = 4
    H, W = 128, 128
    
    for _ in range(3):  # Simulate 3 batches
        logits = torch.randn(batch_size, num_classes, H, W)
        labels = torch.randint(0, num_classes, (batch_size, H, W))
        
        metrics.update(logits, labels)
        metrics.update_boundary(logits, labels, threshold=2)
    
    # Compute all metrics
    results = metrics.compute_all()
    
    print("\nMetrics Results:")
    print(f"  mIoU: {results['mIoU']:.4f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    
    print("\nPer-class IoU:")
    for class_name in class_names:
        key = f'IoU_{class_name}'
        if key in results:
            print(f"  {class_name}: {results[key]:.4f}")
    
    print("\nBoundary Metrics:")
    print(f"  Precision: {results['boundary_precision']:.4f}")
    print(f"  Recall: {results['boundary_recall']:.4f}")
    print(f"  F1-score: {results['boundary_f1']:.4f}")
    
    # Test confusion matrix
    cm = metrics.get_confusion_matrix()
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print(f"Total pixels: {cm.sum()}")
    
    print("\n✓ Metrics test passed!")


def compute_miou(predictions, labels, num_classes=6, ignore_index=-100):
    """
    Compute mean IoU from predictions and labels.
    
    Args:
        predictions: Predicted labels [B, H, W] or [N, H, W]
        labels: Ground truth labels [B, H, W] or [N, H, W]
        num_classes: Number of classes
        ignore_index: Label to ignore
        
    Returns:
        Mean IoU score
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Mask out ignore index
    mask = labels != ignore_index
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        if 0 <= pred < num_classes and 0 <= label < num_classes:
            confusion_matrix[label, pred] += 1
    
    # Compute IoU per class
    iou_per_class = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        if tp + fp + fn == 0:
            continue  # Skip classes not present
        
        iou = tp / (tp + fp + fn)
        iou_per_class.append(iou)
    
    # Return mean IoU
    return np.mean(iou_per_class) if iou_per_class else 0.0
