# UrbanMamba v3: Twin Tower Architecture

**Urban Semantic Segmentation with Symmetric Spatial and Frequency Encoders**

## 🏗️ Architecture Overview

UrbanMamba v3 introduces a **Twin Tower Architecture** that processes RGB and frequency information through symmetric encoders with stage-wise selective fusion.

### Key Features

- **🌐 Twin Symmetric Encoders**: Separate but identical MambaVision backbones for spatial (RGB) and frequency (NSST) streams
- **🔄 Stage-wise Fusion**: MambaFusionBlock at each encoder stage for selective feature integration
- **📊 NSST Transform**: 87-channel frequency decomposition with multi-scale, multi-directional analysis
- **⚡ Efficiency**: ~87% FLOPs reduction compared to processing frequency subbands separately
- **🎯 Clean Design**: Symmetric architecture is more maintainable and extensible

## 📐 Architecture Flow

```
Input RGB [B, 3, H, W]
     |
     ├─────────────────────────────────┐
     │                                  │
     │                            NSST Transform
     │                            (87 subbands)
     │                                  │
     ↓                                  ↓
Spatial Encoder                Frequency Encoder
(MambaVision 3ch)              (MambaVision 87ch)
     │                                  │
[F1, F2, F3, F4]              [F1', F2', F3', F4']
     │                                  │
     └─────────────┬────────────────────┘
                   │
          Stage-wise Fusion
         (MambaFusionBlock)
                   │
       [Fused1, Fused2, Fused3, Fused4]
                   │
       Multi-Scale Aggregation
                   │
          Segmentation Map
            [B, K, H, W]
```

## 🔧 Components

### 1. NSST Decomposition (transforms.py)

Non-Subsampled Shearlet Transform extracts 87 frequency subbands:
- **Scale 1**: 4 directions (2²)
- **Scale 2**: 8 directions (2³)
- **Scale 3**: 16 directions (2⁴)
- **Low-frequency**: 1 component
- **Total**: (4 + 8 + 16 + 1) × 3 RGB channels = **87 channels**

### 2. MambaVision Encoder (encoder.py)

Dual-path encoder architecture:
- **Spatial Encoder**: Processes RGB input (3 channels)
- **Frequency Encoder**: Processes NSST features (87 channels)
- **Variants**: tiny, small, base, large
- **Output**: 4-stage features [F1, F2, F3, F4]

| Variant | Dimensions |
|---------|------------|
| Tiny    | [80, 160, 320, 640] |
| Small   | [96, 192, 384, 768] |
| Base    | [128, 256, 512, 1024] |
| Large   | [160, 320, 640, 1280] |

### 3. MambaFusionBlock (fusion.py)

Selective fusion using Mamba state-space model:
- Concatenates spatial and frequency features
- Projects to original dimension
- Applies Mamba mixer for selective integration
- Residual connection with spatial features
- **Fallback**: Convolutional fusion if mamba-ssm unavailable

### 4. Urban Context Decoder (decoder.py)

Progressive upsampling with dual attention for urban boundary preservation:

**Spatial Attention**:
- 7×7 convolution for large receptive field
- Captures spatial context and boundaries

**Channel Attention**:
- Global Average + Max Pooling
- Shared MLP for channel recalibration

**Urban Context Block (UCB)**:
- Combines spatial + channel attention
- Refines features for urban structures

**Decoder Architecture**:
- 4 progressive upsampling stages
- Skip connections from fusion blocks
- UCB at each stage for refinement
- Final 4× upsampling to original resolution

**Parameters**: 4.9M (22.5% of total)

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision
pip install timm==1.0.15
pip install mamba-ssm==2.2.4  # Optional, for MambaFusionBlock
pip install pyyaml tqdm
```

### Demo

```bash
python demo.py
```

This runs 5 demonstrations:
1. Simple inference
2. Training step with loss computation
3. NSST feature extraction
4. Architecture inspection
5. Twin tower data flow

### Training

1. **Prepare your dataset** (implement dataset class)
2. **Update configuration**:
   ```yaml
   # configs/config.yaml
   model:
     num_classes: 6
     variant: 'tiny'
   
   data:
     train_dir: 'path/to/train'
     val_dir: 'path/to/val'
   ```

3. **Train the model**:
   ```bash
   python train.py --config configs/config.yaml
   ```

## 📊 Model Variants

| Variant | Parameters | FLOPs | mIoU* | Decoder |
|---------|-----------|-------|-------|---------|
| Tiny    | 21.6M     | ~45G  | TBD   | 4.9M (22.5%) |
| Small   | ~32M      | ~70G  | TBD   | ~6M |
| Base    | ~54M      | ~120G | TBD   | ~8M |
| Large   | ~82M      | ~200G | TBD   | ~10M |

*Results to be updated after training

**Parameter Breakdown (Tiny)**:
- NSST Extractor: 8K (0.0%)
- Spatial Encoder: 7.4M (34.2%)
- Frequency Encoder: 7.7M (35.7%)
- Fusion Blocks: 1.7M (7.6%)
- Urban Context Decoder: 4.9M (22.5%)

## 🎯 Usage Example

```python
import torch
from models import create_urban_mamba_v3

# Create model
model = create_urban_mamba_v3(
    num_classes=6,
    variant='tiny',
    pretrained_spatial=None
)

# Inference
rgb = torch.randn(1, 3, 512, 512)
output = model(rgb)  # [1, 6, 512, 512]

# Extract NSST features
nsst_features = model.extract_nsst_features(rgb)  # [1, 87, 512, 512]
```

## 📂 Project Structure

```
urban_mamba/
├── models/
│   ├── __init__.py
│   ├── model.py          # Main UrbanMambaV3 model
│   ├── encoder.py        # MambaVision encoders
│   ├── fusion.py         # MambaFusionBlock
│   ├── decoder.py        # Urban Context Decoder
│   └── transforms.py     # NSST decomposition
├── losses/
│   ├── __init__.py
│   ├── loss.py           # CompositeLoss
│   └── metrics.py        # SegmentationMetrics
├── configs/
│   └── config.yaml       # Training configuration
├── train.py              # Training script
└── demo.py               # Quick start demo
```

## 🔬 Key Innovations

### 1. Twin Tower Symmetry
Unlike asymmetric dual-branch designs, v3 uses **identical encoders** for both streams:
- Easier to train and tune
- More balanced feature learning
- Cleaner architecture

### 2. Stage-wise Fusion
Fusion occurs at **each encoder stage** rather than just at the end:
- Better feature alignment across modalities
- Progressive information integration
- Reduced information loss

### 3. Selective Integration with Mamba
MambaFusionBlock uses **state-space models** to selectively combine features:
- Learns which modality (spatial vs frequency) is more important
- Adaptive fusion based on input content
- More powerful than simple concatenation or addition

### 4. Efficiency
Processing NSST as a single 87-channel tensor instead of 29 separate subbands:
- **~87% reduction in FLOPs**
- Faster training and inference
- Lower memory footprint

## 📝 Citation

```bibtex
@article{urbanmamba_v3,
  title={UrbanMamba v3: Twin Tower Architecture for Urban Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for urban scene understanding**

