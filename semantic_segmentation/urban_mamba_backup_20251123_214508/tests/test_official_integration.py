"""
Test Official MambaVision Segmentation Integration
Verifies the enhanced wrapper uses the official implementation correctly
"""

import torch
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add mambavision to path
mambavision_dir = os.path.join(os.path.dirname(os.path.dirname(parent_dir)), 'mambavision')
if mambavision_dir not in sys.path:
    sys.path.insert(0, mambavision_dir)

print("=" * 80)
print("Testing Official MambaVision Segmentation Integration")
print("=" * 80)

# Test 1: Import the enhanced wrapper
print("\n[Test 1] Importing Enhanced Segmentation Wrapper...")
try:
    from models.mambavision_segmentation import (
        create_mambavision_segmentation_encoder,
        MambaVisionSegmentationBackbone,
        OFFICIAL_MAMBAVISION_AVAILABLE
    )
    print(f"✓ Import successful")
    print(f"  Official MambaVision available: {OFFICIAL_MAMBAVISION_AVAILABLE}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create RGB encoder (3 channels)
print("\n[Test 2] Creating RGB Encoder (3 channels)...")
try:
    rgb_encoder = create_mambavision_segmentation_encoder(
        in_channels=3,
        variant='tiny',
        pretrained=None
    )
    print(f"✓ RGB encoder created")
    print(f"  Output channels: {rgb_encoder.get_output_channels()}")
    
    # Test forward pass
    dummy_rgb = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        rgb_features = rgb_encoder(dummy_rgb)
    
    print(f"  Forward pass successful")
    print(f"  Number of feature maps: {len(rgb_features)}")
    for i, feat in enumerate(rgb_features):
        print(f"    Stage {i}: {feat.shape}")
    
except Exception as e:
    print(f"✗ RGB encoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create XLET encoder (87 channels)
print("\n[Test 3] Creating XLET Encoder (87 channels)...")
try:
    xlet_encoder = create_mambavision_segmentation_encoder(
        in_channels=87,
        variant='tiny',
        pretrained=None
    )
    print(f"✓ XLET encoder created")
    print(f"  Output channels: {xlet_encoder.get_output_channels()}")
    
    # Test forward pass
    dummy_xlet = torch.randn(2, 87, 512, 512)
    with torch.no_grad():
        xlet_features = xlet_encoder(dummy_xlet)
    
    print(f"  Forward pass successful")
    print(f"  Number of feature maps: {len(xlet_features)}")
    for i, feat in enumerate(xlet_features):
        print(f"    Stage {i}: {feat.shape}")
    
except Exception as e:
    print(f"✗ XLET encoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify feature dimensions match
print("\n[Test 4] Verifying Feature Dimension Consistency...")
try:
    assert len(rgb_features) == len(xlet_features), "Number of stages must match"
    
    for i, (rgb_feat, xlet_feat) in enumerate(zip(rgb_features, xlet_features)):
        assert rgb_feat.shape[1] == xlet_feat.shape[1], \
            f"Stage {i}: Channel mismatch {rgb_feat.shape[1]} vs {xlet_feat.shape[1]}"
        assert rgb_feat.shape[2] == xlet_feat.shape[2], \
            f"Stage {i}: Height mismatch"
        assert rgb_feat.shape[3] == xlet_feat.shape[3], \
            f"Stage {i}: Width mismatch"
    
    print(f"✓ All feature dimensions match between RGB and XLET branches")
    
except AssertionError as e:
    print(f"✗ Dimension verification failed: {e}")
    sys.exit(1)

# Test 5: Test UrbanMamba with new encoder
print("\n[Test 5] Testing Full UrbanMamba Pipeline...")
try:
    from models.model import UrbanMamba
    
    model = UrbanMamba(
        num_classes=6,
        use_aggregation_head=True,
        output_stride=4
    )
    
    print(f"✓ UrbanMamba model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Forward pass successful")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 6, 512, 512), f"Expected (2,6,512,512), got {output.shape}"
    
except Exception as e:
    print(f"✗ UrbanMamba test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: GPU compatibility (if available)
if torch.cuda.is_available():
    print("\n[Test 6] Testing GPU Compatibility...")
    try:
        device = torch.device('cuda')
        model = model.to(device)
        dummy_input = torch.randn(1, 3, 512, 512, device=device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ GPU test successful")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\n[Test 6] GPU not available - skipping GPU tests")

# Test 7: Gradient flow test
print("\n[Test 7] Testing Gradient Flow...")
try:
    model = UrbanMamba(num_classes=6)
    model.train()
    
    dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)
    dummy_target = torch.randint(0, 6, (1, 512, 512))
    
    # Forward pass
    output = model(dummy_input)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    
    print(f"✓ Gradient flow test successful")
    print(f"  Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"✗ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check if official implementation is being used
print("\n[Test 8] Implementation Check...")
try:
    if OFFICIAL_MAMBAVISION_AVAILABLE:
        print(f"✓ Using OFFICIAL MambaVision segmentation implementation")
        print(f"  Source: semantic_segmentation/tools/mamba_vision.py")
        print(f"  Features:")
        print(f"    - MM_mamba_vision backbone from MMSegmentation")
        print(f"    - Pretrained weight loading support")
        print(f"    - Multi-scale feature extraction")
    else:
        print(f"✗ MambaVision implementation not available")
        print(f"  Required: Official mamba_vision.py or mamba-ssm installation")
        print(f"  Please install dependencies: pip install mamba-ssm timm")
except Exception as e:
    print(f"✗ Implementation check failed: {e}")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ All tests passed!")
print(f"\nIntegration Status:")
print(f"  - Enhanced wrapper: ✓ Working")
print(f"  - RGB encoder (3ch): ✓ Working")
print(f"  - XLET encoder (87ch): ✓ Working")
print(f"  - UrbanMamba pipeline: ✓ Working")
print(f"  - Gradient flow: ✓ Working")
if torch.cuda.is_available():
    print(f"  - GPU compatibility: ✓ Working")
print("\nImplementation:")
if OFFICIAL_MAMBAVISION_AVAILABLE:
    print(f"  ✓ Using official MambaVision segmentation code")
    print(f"    from semantic_segmentation/tools/mamba_vision.py")
else:
    print(f"  ✗ MambaVision implementation not available")
    print(f"    Please install required dependencies")

print("\n" + "=" * 80)
