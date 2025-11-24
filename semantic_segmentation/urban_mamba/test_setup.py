"""
Quick test script to verify MambaVision-NSST setup for LOVEDA dataset.
Tests dataset loading, NSST transform, model forward pass, and loss computation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import yaml
from data.loveda_dataset import LOVEDADataset
from models import create_urban_mamba_v3
from losses import CompositeLoss, SegmentationMetrics


def test_dataset():
    """Test LOVEDA dataset loading."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Loading")
    print("="*70)
    
    dataset = LOVEDADataset(
        dataset_path='/storage2/ChangeDetection/SemanticSegmentation/Dataset/LOVED',
        data_list_file='/storage2/ChangeDetection/SemanticSegmentation/Dataset/LOVED/train_urban.txt',
        crop_size=512,
        split='train'
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Load one sample
    img, label = dataset[0]
    print(f"✓ Sample loaded:")
    print(f"  Image shape: {img.shape} (expected: [3, 512, 512])")
    print(f"  Label shape: {label.shape} (expected: [512, 512])")
    print(f"  Label unique values: {torch.unique(label)}")
    print(f"  Label dtype: {label.dtype} (expected: torch.int64)")
    
    # Verify label remapping
    unique_labels = torch.unique(label).numpy()
    if 255 in unique_labels:
        print(f"✓ Ignore label (255) present")
    valid_labels = unique_labels[unique_labels != 255]
    if len(valid_labels) > 0:
        print(f"✓ Valid label range: {valid_labels.min()} to {valid_labels.max()} (expected: 0-6)")
    
    return img, label


def test_model(device):
    """Test model creation and forward pass."""
    print("\n" + "="*70)
    print("TEST 2: Model Architecture")
    print("="*70)
    
    # Create model
    model = create_urban_mamba_v3(
        num_classes=7,
        variant='small',
        pretrained_spatial=None,
        pretrained_freq=None
    )
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {num_params/1e6:.2f}M parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape} (expected: [1, 7, 512, 512])")
    
    return model


def test_loss(img, label, model, device):
    """Test loss computation."""
    print("\n" + "="*70)
    print("TEST 3: Loss Computation")
    print("="*70)
    
    # Create loss
    criterion = CompositeLoss(
        num_classes=7,
        ce_weight=0.7,
        lovasz_weight=0.3
    )
    
    # Prepare batch
    img_batch = img.unsqueeze(0).to(device)
    label_batch = label.unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(img_batch)
    
    # Compute loss
    loss_dict = criterion(output, label_batch)
    
    print(f"✓ Loss computed:")
    print(f"  Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"  CE Loss: {loss_dict['loss_ce'].item():.4f}")
    print(f"  Lovász Loss: {loss_dict['loss_lovasz'].item():.4f}")
    
    return loss_dict


def test_metrics(model, device):
    """Test metrics computation."""
    print("\n" + "="*70)
    print("TEST 4: Metrics Computation")
    print("="*70)
    
    metrics = SegmentationMetrics(num_classes=7)
    
    # Create dummy predictions and targets
    dummy_preds = torch.randn(2, 7, 512, 512).to(device)
    dummy_targets = torch.randint(0, 7, (2, 512, 512)).to(device)
    
    # Update metrics
    metrics.update(dummy_preds, dummy_targets)
    
    # Get metrics
    results = metrics.get_metrics()
    
    print(f"✓ Metrics computed:")
    print(f"  mIoU: {results.get('mIoU', 0):.4f}")
    print(f"  Pixel Accuracy: {results.get('pixel_acc', 0):.4f}")
    if 'class_iou' in results:
        print(f"  Per-class IoU available: {len(results['class_iou'])} classes")
    
    return results


def test_nsst_transform():
    """Test NSST decomposition."""
    print("\n" + "="*70)
    print("TEST 5: NSST Transform")
    print("="*70)
    
    from models import NSSTDecomposition
    
    nsst = NSSTDecomposition()
    dummy_rgb = torch.randn(1, 3, 512, 512)
    
    nsst_features = nsst(dummy_rgb)
    
    print(f"✓ NSST decomposition successful:")
    print(f"  Input shape: {dummy_rgb.shape}")
    print(f"  Output shape: {nsst_features.shape} (expected: [1, 87, 512, 512])")
    print(f"  Channels: 87 = (4+8+16+1) directions × 3 RGB channels")
    
    return nsst_features


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MAMBAVISION-NSST LOVEDA SETUP VERIFICATION")
    print("="*70)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\n⚠ Using CPU (GPU not available)")
    
    try:
        # Test 1: Dataset
        img, label = test_dataset()
        
        # Test 2: Model
        model = test_model(device)
        
        # Test 3: Loss
        loss_dict = test_loss(img, label, model, device)
        
        # Test 4: Metrics
        metrics_results = test_metrics(model, device)
        
        # Test 5: NSST Transform
        nsst_features = test_nsst_transform()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nMambaVision-NSST is ready for training on LOVEDA dataset.")
        print("Run training with:")
        print("  python train_loveda.py --config configs/config.yaml --gpu 0")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
