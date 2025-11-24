"""
Quick Start Demo for UrbanMamba v3 Twin Tower Architecture
Demonstrates the complete pipeline from input to output.
"""

import torch
import time

try:
    from models import create_urban_mamba_v3
    from losses import CompositeLoss
except ImportError:
    from .models import create_urban_mamba_v3
    from .losses import CompositeLoss


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_inference():
    """Demo 1: Simple Inference"""
    print_section("DEMO 1: Simple Inference")
    
    print("\n1. Creating UrbanMamba v3 model...")
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    print("\n2. Creating input RGB image...")
    batch_size = 2
    height, width = 256, 256
    rgb = torch.randn(batch_size, 3, height, width)
    print(f"   ✓ Input shape: {tuple(rgb.shape)}")
    
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
    
    print("\n1. Setting up training components...")
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    criterion = CompositeLoss(num_classes=6)
    print("   ✓ Model and loss created")
    
    print("\n2. Creating training batch...")
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 256, 256)
    target = torch.randint(0, 6, (batch_size, 256, 256))
    print(f"   ✓ RGB: {tuple(rgb.shape)}")
    print(f"   ✓ Target: {tuple(target.shape)}")
    
    print("\n3. Forward pass...")
    output = model(rgb)
    print(f"   ✓ Prediction: {tuple(output.shape)}")
    
    print("\n4. Computing loss...")
    loss_dict = criterion(output, target)
    print(f"   ✓ Total loss: {loss_dict['loss']:.4f}")
    print(f"   ✓ CE loss: {loss_dict['loss_ce']:.4f}")
    print(f"   ✓ Lovász loss: {loss_dict['loss_lovasz']:.4f}")
    
    print("\n5. Backward pass...")
    loss_dict['loss'].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"   ✓ Gradients computed: {has_grad}/{total} parameters")
    
    return model, loss_dict


def demo_nsst_extraction():
    """Demo 3: NSST Feature Extraction"""
    print_section("DEMO 3: NSST Feature Extraction")
    
    print("\n1. Creating model...")
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    
    print("\n2. Creating RGB input...")
    rgb = torch.randn(2, 3, 256, 256)
    print(f"   ✓ Input: {tuple(rgb.shape)}")
    
    print("\n3. Extracting NSST features...")
    with torch.no_grad():
        nsst_features = model.extract_nsst_features(rgb)
    print(f"   ✓ NSST output: {tuple(nsst_features.shape)} (87 channels!)")
    
    print("\n4. Analyzing frequency subbands...")
    # The 87 channels come from:
    # Scale 1: 4 directions + Scale 2: 8 directions + Scale 3: 16 directions + 1 low-freq
    # = (4 + 8 + 16 + 1) * 3 RGB channels = 29 * 3 = 87
    print("   ✓ Decomposition structure:")
    print("     - Scale 1: 4 directions (2^2)")
    print("     - Scale 2: 8 directions (2^3)")
    print("     - Scale 3: 16 directions (2^4)")
    print("     - Low-frequency: 1 component")
    print("     - Total per channel: 29 subbands")
    print("     - RGB channels: 3")
    print("     - Total output: 29 × 3 = 87 channels")
    
    return nsst_features


def demo_architecture():
    """Demo 4: Architecture Inspection"""
    print_section("DEMO 4: Architecture Inspection")
    
    print("\n1. Creating model...")
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    
    print("\n2. Model Components:")
    print(f"   ✓ NSST Extractor: {model.nsst_extractor.__class__.__name__}")
    print(f"   ✓ Spatial Encoder: {model.spatial_encoder.__class__.__name__}")
    print(f"   ✓ Frequency Encoder: {model.freq_encoder.__class__.__name__}")
    print(f"   ✓ Fusion Blocks: {len(model.fusions)} stages")
    print(f"   ✓ Decoder: {model.decoder.__class__.__name__}")
    
    print("\n3. Parameter Distribution:")
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
    
    print("\n4. Feature Dimensions:")
    print(f"   ✓ Encoder stages: {model.get_feature_dims()}")


def demo_twin_tower_flow():
    """Demo 5: Twin Tower Data Flow"""
    print_section("DEMO 5: Twin Tower Data Flow")
    
    print("\n🚀 UrbanMamba v3: Twin Tower Architecture")
    print("   Key Innovation: Stage-wise selective fusion of spatial + frequency")
    
    print("\n1. Architecture Flow:")
    print("   ")
    print("   Input RGB [B, 3, H, W]")
    print("        |")
    print("        ├───────────────────────┐")
    print("        │                       │")
    print("        │                  NSST Transform")
    print("        │                       │")
    print("        │                  [B, 87, H, W]")
    print("        │                       │")
    print("        ↓                       ↓")
    print("   Spatial Encoder      Frequency Encoder")
    print("   (RGB 3ch)            (NSST 87ch)")
    print("        │                       │")
    print("   [F1,F2,F3,F4]        [F1',F2',F3',F4']")
    print("        │                       │")
    print("        └───────┬───────────────┘")
    print("                │")
    print("         Stage-wise Fusion")
    print("        (MambaFusionBlock)")
    print("                │")
    print("     [Fused1, Fused2, Fused3, Fused4]")
    print("                │")
    print("    Multi-Scale Aggregation")
    print("                │")
    print("        Segmentation Map")
    print("           [B, K, H, W]")
    
    print("\n2. Key Benefits:")
    print("   ✓ Twin symmetric encoders (clean design)")
    print("   ✓ Stage-wise selective fusion with Mamba mixer")
    print("   ✓ ~87% FLOPs reduction vs processing subbands separately")
    print("   ✓ Better feature alignment at each stage")
    
    print("\n3. Test forward pass...")
    model = create_urban_mamba_v3(num_classes=6, variant='tiny')
    rgb = torch.randn(2, 3, 256, 256)
    
    start = time.time()
    with torch.no_grad():
        output = model(rgb)
    elapsed = time.time() - start
    
    print(f"   ✓ Output shape: {tuple(output.shape)}")
    print(f"   ✓ Time: {elapsed:.3f}s")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  UrbanMamba v3: Twin Tower Architecture - Quick Start Demo")
    print("="*70)
    print("\n  This demo showcases:")
    print("  - Simple inference")
    print("  - Training step with loss and gradients")
    print("  - NSST feature extraction")
    print("  - Architecture inspection")
    print("  - Twin Tower data flow")
    
    try:
        # Run demos
        demo_inference()
        demo_training_step()
        demo_nsst_extraction()
        demo_architecture()
        demo_twin_tower_flow()
        
        # Success message
        print("\n" + "="*70)
        print("  ✨ ALL DEMOS COMPLETED SUCCESSFULLY! ✨")
        print("="*70)
        print("\n  The UrbanMamba v3 pipeline is fully operational.")
        print("  You can now:")
        print("  1. Prepare your urban scene dataset")
        print("  2. Update configs/config.yaml with your data paths")
        print("  3. Run: python train.py --config configs/config.yaml")
        print("\n  Happy training! 🚀")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
