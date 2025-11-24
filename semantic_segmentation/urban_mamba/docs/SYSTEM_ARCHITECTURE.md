# UrbanMamba v3: System Architecture Documentation

## Overview

UrbanMamba v3 implements a **Twin Tower Architecture** for urban semantic segmentation with:
- **Twin Symmetric Encoders**: Spatial (RGB) + Frequency (NSST)
- **Stage-wise Fusion**: MambaFusionBlock at each encoder stage
- **Urban Context Decoder**: Progressive upsampling with dual attention
- **87-Channel NSST**: Multi-scale frequency decomposition
- **~87% FLOPs Reduction**: Efficient frequency processing

## Complete Architecture Diagram

```
Input RGB [B, 3, H, W]
         │
         ├─────────────────────────────────┐
         │                                  │
         │                            ┌─────▼─────┐
         │                            │   NSST    │
         │                            │ Transform │
         │                            │ 3→87 ch   │
         │                            └─────┬─────┘
         │                                  │
         │                          [B, 87, H, W]
         │                                  │
    ┌────▼────┐                      ┌─────▼─────┐
    │ Spatial │                      │ Frequency │
    │ Encoder │                      │  Encoder  │
    │ (3 ch)  │                      │  (87 ch)  │
    │MambaVis │                      │ MambaVis  │
    └────┬────┘                      └─────┬─────┘
         │                                  │
    Stage 1: [80, H/4, W/4]      Stage 1: [80, H/4, W/4]
         ├─────────────────┬─────────────────┤
         │          MambaFusionBlock         │
         └───────────────┬───────────────────┘
                  [80, H/4, W/4] Fused1
         
    Stage 2: [160, H/8, W/8]     Stage 2: [160, H/8, W/8]
         ├─────────────────┬─────────────────┤
         │          MambaFusionBlock         │
         └───────────────┬───────────────────┘
                  [160, H/8, W/8] Fused2
         
    Stage 3: [320, H/16, W/16]   Stage 3: [320, H/16, W/16]
         ├─────────────────┬─────────────────┤
         │          MambaFusionBlock         │
         └───────────────┬───────────────────┘
                 [320, H/16, W/16] Fused3
         
    Stage 4: [640, H/32, W/32]   Stage 4: [640, H/32, W/32]
         ├─────────────────┬─────────────────┤
         │          MambaFusionBlock         │
         └───────────────┬───────────────────┘
                 [640, H/32, W/32] Fused4
                         │
              ┌──────────▼──────────┐
              │  Urban Context      │
              │     Decoder         │
              │  (4 stages + UCB)   │
              └──────────┬──────────┘
                         │
                  Segmentation Map
                    [B, 6, H, W]
```

## Component Details

### 1. NSST Transform (87 channels)

```
Scale 1: 4 directions (2²)    → 4 subbands × 3 RGB = 12 channels
Scale 2: 8 directions (2³)    → 8 subbands × 3 RGB = 24 channels
Scale 3: 16 directions (2⁴)   → 16 subbands × 3 RGB = 48 channels
Low-freq: 1 component         → 1 subband × 3 RGB = 3 channels
Total: (4+8+16+1) × 3 = 87 channels
```

### 2. Twin Encoders (Symmetric)

| Variant | Dimensions | Params (each) |
|---------|------------|---------------|
| Tiny    | [80, 160, 320, 640] | 7.4M |
| Small   | [96, 192, 384, 768] | 11M |
| Base    | [128, 256, 512, 1024] | 18M |
| Large   | [160, 320, 640, 1280] | 28M |

### 3. MambaFusionBlock

```python
def forward(spatial_feat, freq_feat):
    # 1. Concatenate
    concat = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 2C, H, W]
    
    # 2. Flatten & Project
    x = concat.flatten(2).transpose(1, 2)  # [B, H*W, 2C]
    x = self.proj(x)  # [B, H*W, C]
    
    # 3. Mamba Mixer (Selective State-Space Model)
    x = self.mixer(x)  # [B, H*W, C]
    
    # 4. Normalize & Reshape
    x = self.norm(x)  # [B, H*W, C]
    x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
    
    # 5. Residual Connection
    return x + spatial_feat
```

### 4. Urban Context Decoder

**Components**:
- **Spatial Attention**: 7×7 conv for large receptive field
- **Channel Attention**: GAP + GMP with MLP
- **Urban Context Block (UCB)**: Dual attention + refinement
- **Progressive Upsampling**: 4 stages with skip connections

**Decoder Flow**:
```
F4_fused [640, H/32, W/32]
    │
    ├─ UCB → D1 [320, H/32, W/32]
    │
    ├─ Upsample(2×) + F3_fused → UCB → D2 [160, H/16, W/16]
    │
    ├─ Upsample(2×) + F2_fused → UCB → D3 [80, H/8, W/8]
    │
    ├─ Upsample(2×) + F1_fused → UCB → D4 [40, H/4, W/4]
    │
    └─ Upsample(4×) → Classifier → Output [6, H, W]
```

## Model Statistics (Tiny Variant)

### Parameters

| Component | Parameters | % Total |
|-----------|-----------|---------|
| NSST Extractor | 8K | 0.0% |
| Spatial Encoder | 7.4M | 34.2% |
| Frequency Encoder | 7.7M | 35.7% |
| Fusion Blocks (4×) | 1.7M | 7.6% |
| Urban Context Decoder | 4.9M | 22.5% |
| **Total** | **21.6M** | **100%** |

### Computational Cost

- **FLOPs**: ~45G (512×512 input)
- **GPU Memory**: ~6GB (batch=8)
- **Inference Time**: ~25ms/image (RTX 3090)

## Training Configuration

```yaml
model:
  num_classes: 6
  variant: 'tiny'
  
training:
  batch_size: 8
  lr: 0.0001
  optimizer: AdamW
  weight_decay: 0.01
  epochs: 100
  scheduler: CosineAnnealingLR
  
loss:
  type: Composite
  ce_weight: 0.7
  lovasz_weight: 0.3
```

## Usage

```python
from models import create_urban_mamba_v3

# Create model
model = create_urban_mamba_v3(
    num_classes=6,
    variant='tiny'
)

# Forward pass
rgb = torch.randn(2, 3, 512, 512)
output = model(rgb)  # [2, 6, 512, 512]

# Extract NSST
nsst = model.extract_nsst_features(rgb)  # [2, 87, 512, 512]
```

## Key Benefits

1. **Twin Symmetry**: Easier to train and tune than asymmetric designs
2. **Stage-wise Fusion**: Better feature alignment across modalities
3. **Selective Integration**: Mamba learns which modality to emphasize
4. **Urban Context Decoder**: Preserves boundaries and thin structures
5. **Efficiency**: 87% FLOPs reduction vs naive subband processing

## File Structure

```
urban_mamba/
├── models/
│   ├── model.py          # UrbanMambaV3
│   ├── transforms.py     # NSSTDecomposition
│   ├── encoder.py        # MambaVisionEncoder
│   ├── fusion.py         # MambaFusionBlock
│   └── decoder.py        # UrbanContextDecoder
├── losses/
│   ├── loss.py           # CompositeLoss
│   └── metrics.py        # SegmentationMetrics
├── train.py
├── demo.py
└── configs/config.yaml
```

