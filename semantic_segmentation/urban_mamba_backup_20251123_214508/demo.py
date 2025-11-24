#!/usr/bin/env python
"""
Quick Start Demo for UrbanMamba v3 with Twin Tower Architecture
Demonstrates the complete pipeline from input to output
"""

import torch
import torch.nn as nn
from models.model import UrbanMamba, create_urban_mamba
from losses.lovasz_loss import CompositeLoss
import time

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def demo_inference():
    """Demo 1: Simple Inference"""
    print_section("DEMO 1: Simple Inference (UrbanMamba v3)")
    
    # Create model
    print("\n1. Creating UrbanMamba v3 model (Twin Tower)...")
    model = create_urban_mamba(num_classes=6, variant='tiny')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    # Create dummy input
    print("\n2. Creating input RGB image...")
    batch_size = 2
    height, width = 256, 256
    rgb = torch.randn(batch_size, 3, height, width)
    print(f"   ✓ Input shape: {tuple(rgb.shape)}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    start = time.time()
    with torch.no_grad():
        output = model(rgb)
    elapsed = time.time() - start
    
    print(f"   ✓ Output shape: {tuple(output.shape)}")
    print(f"   ✓ Time: {elapsed:.3f}s ({elapsed/batch_size:.3f}s per image)")
    print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return model, output

def demo_training_step():
    """Demo 2: Training Step"""
    print_section("DEMO 2: Training Step")
    
    # Create model and loss
    print("\n1. Setting up training components...")
    model = create_urban_mamba(num_classes=6, variant='tiny')
    criterion = CompositeLoss(num_classes=6)
    print("   ✓ Model and loss created")
    
    # Create dummy data
    print("\n2. Creating training batch...")
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 256, 256)
    target = torch.randint(0, 6, (batch_size, 256, 256))
    print(f"   ✓ RGB: {tuple(rgb.shape)}")
    print(f"   ✓ Target: {tuple(target.shape)}")
    
    # Forward pass
    print("\n3. Forward pass...")
    output = model(rgb)
    print(f"   ✓ Prediction: {tuple(output.shape)}")
    
    # Compute loss
    print("\n4. Computing loss...")
    loss_dict = criterion(output, target)
    print(f"   ✓ Total loss: {loss_dict['loss']:.4f}")
    print(f"   ✓ CE loss: {loss_dict['loss_ce']:.4f}")
    print(f"   ✓ Lovasz loss: {loss_dict['loss_lovasz']:.4f}")
    
    # Backward pass
    print("\n5. Backward pass...")
    loss_dict['loss'].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"   ✓ Gradients computed: {has_grad}/{total} parameters")
    
    return model, loss_dict

def demo_dual_branch():
    """Demo 3: Twin Tower Feature Extraction"""
    print_section("DEMO 3: Twin Tower Feature Extraction")
    
    from models.transforms import NSSTDecomposition
    from models.mambavision_segmentation import create_mambavision_segmentation_encoder
    
    # Create components
    print("\n1. Creating NSST transformer...")
    nsst = NSSTDecomposition(scales=3, directions_profile=[2, 3, 4])
    print("   ✓ NSST created (3 scales, directions=[2,3,4])")
    
    print("\n2. Creating spatial encoder (RGB - 3 channels)...")
    spatial_encoder = create_mambavision_segmentation_encoder(in_channels=3, variant='tiny', pretrained=None)
    print(f"   ✓ Output channels: {spatial_encoder.get_output_channels()}")
    
    print("\n3. Creating frequency encoder (NSST - 87 channels)...")
    freq_encoder = create_mambavision_segmentation_encoder(in_channels=87, variant='tiny', pretrained=None)
    print(f"   ✓ Output channels: {freq_encoder.get_output_channels()}")
    
    # Process input
    print("\n4. Processing RGB image...")
    rgb = torch.randn(1, 3, 256, 256)
    print(f"   ✓ Input: {tuple(rgb.shape)}")
    
    print("\n5. Extracting NSST features...")
    with torch.no_grad():
        freq_features = nsst(rgb)
    print(f"   ✓ NSST output: {tuple(freq_features.shape)} (87 channels!)")
    
    print("\n6. Extracting spatial features...")
    with torch.no_grad():
        spatial_feats = spatial_encoder(rgb)
    print("   ✓ Spatial features:")
    for i, feat in enumerate(spatial_feats):
        print(f"      Stage {i+1}: {tuple(feat.shape)}")
    
    print("\n7. Extracting frequency features...")
    with torch.no_grad():
        freq_feats = freq_encoder(freq_features)
    print("   ✓ Frequency features:")
    for i, feat in enumerate(freq_feats):
        print(f"      Stage {i+1}: {tuple(feat.shape)}")
    
    print("\n8. Stage-wise fusion with MambaFusionBlock...")
    from models.mamba_modules import MambaFusionBlock
    fusion = MambaFusionBlock(channels=80)
    with torch.no_grad():
        fused = fusion(spatial_feats[0], freq_feats[0])
    print(f"   ✓ Fused features: {tuple(fused.shape)}")

def demo_architecture():
    """Demo 4: Architecture Inspection"""
    print_section("DEMO 4: Architecture Inspection")
    
    model = create_urban_mamba(num_classes=6, variant='tiny')
    
    print("\n1. Model Components:")
    print(f"   ✓ NSST Extractor: {model.nsst_extractor.__class__.__name__}")
    print(f"   ✓ Spatial Encoder: {model.spatial_encoder.__class__.__name__}")
    print(f"   ✓ Frequency Encoder: {model.freq_encoder.__class__.__name__}")
    print(f"   ✓ Fusion Blocks: {len(model.fusions)} stages")
    print(f"   ✓ Decoder: {model.decoder.__class__.__name__}")
    
    print("\n2. Parameter Distribution:")
    nsst_params = sum(p.numel() for p in model.nsst_extractor.parameters())
    spatial_params = sum(p.numel() for p in model.spatial_encoder.parameters())
    freq_params = sum(p.numel() for p in model.freq_encoder.parameters())
    fusion_params = sum(p.numel() for m in model.fusions for p in m.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total = nsst_params + spatial_params + freq_params + fusion_params + decoder_params
    
    print(f"   ✓ NSST: {nsst_params:,} ({100*nsst_params/total:.1f}%)")
    print(f"   ✓ Spatial Encoder: {spatial_params:,} ({100*spatial_params/total:.1f}%)")
    print(f"   ✓ Frequency Encoder: {freq_params:,} ({100*freq_params/total:.1f}%)")
    print(f"   ✓ Fusion: {fusion_params:,} ({100*fusion_params/total:.1f}%)")
    print(f"   ✓ Decoder: {decoder_params:,} ({100*decoder_params/total:.1f}%)")
    print(f"   ✓ TOTAL: {total:,}")
    
    print("\n3. Encoder Dimensions:")
    print(f"   ✓ {model.dims}")

def demo_v3_twin_tower():
    """Demo 5: UrbanMamba v3 Twin Tower Architecture"""
    print_section("DEMO 5: UrbanMamba v3 - Twin Tower Architecture")
    
    print("\n🚀 UrbanMamba v3: Twin Tower with MambaFusion")
    print("   Key Innovation: Stage-wise selective fusion of spatial + frequency")
    
    # Create v3 model
    print("\n1. Creating UrbanMamba v3 model...")
    model = create_urban_mamba(num_classes=6, variant='tiny')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    # Create input
    print("\n2. Creating input RGB image...")
    rgb = torch.randn(2, 3, 256, 256)
    print(f"   ✓ Input shape: {tuple(rgb.shape)}")
    
    # Extract NSST features
    print("\n3. Extracting NSST features...")
    nsst_features = model.extract_nsst_features(rgb)
    print(f"   ✓ NSST shape: {tuple(nsst_features.shape)} (87 channels!)")
    
    # Forward pass
    print("\n4. Running twin tower forward pass...")
    start = time.time()
    with torch.no_grad():
        output = model(rgb)
    elapsed = time.time() - start
    
    print(f"   ✓ Output shape: {tuple(output.shape)}")
    print(f"   ✓ Time: {elapsed:.3f}s")
    
    # Architecture details
    print("\n5. v3 Architecture Details:")
    print(f"   ✓ Spatial Encoder: 3 channels → {model.dims}")
    print(f"   ✓ Frequency Encoder: 87 channels → {model.dims}")
    print(f"   ✓ Fusion Blocks: {len(model.fusions)} stages")
    print(f"   ✓ Each fusion uses MambaFusionBlock (selective integration)")
    
    # Architecture benefits
    print("\n6. v3 Key Benefits:")
    print("   ✓ Twin symmetric encoders (clean design)")
    print("   ✓ Stage-wise selective fusion with Mamba mixer")
    print("   ✓ ~87% FLOPs reduction vs processing subbands separately")
    print("   ✓ Better feature alignment at each stage")
    
    return model, output

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  UrbanMamba v3: Twin Tower Architecture Demo")
    print("="*70)
    print("\n  This demo showcases the complete v3 pipeline:")
    print("  - Simple inference")
    print("  - Training step with loss and gradients")
    print("  - Twin tower feature extraction")
    print("  - Architecture inspection")
    print("  - Twin Tower v3 architecture details")
    
    try:
        # Run demos
        demo_inference()
        demo_training_step()
        demo_dual_branch()
        demo_architecture()
        demo_v3_twin_tower()
        
        # Success message
        print("\n" + "="*70)
        print("  ✨ ALL DEMOS COMPLETED SUCCESSFULLY! ✨")
        print("="*70)
        print("\n  The UrbanMamba v3 pipeline is fully operational.")
        print("  You can now:")
        print("  1. Prepare your urban scene dataset")
        print("  2. Update config.yaml with your data paths")
        print("  3. Run: python train.py")
        print("\n  Happy training! 🚀")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
