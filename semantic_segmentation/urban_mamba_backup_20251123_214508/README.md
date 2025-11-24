# UrbanMamba: Dual-Branch Architecture with XLET-NSST for Urban Semantic Segmentation

## Overview

UrbanMamba is a state-of-the-art dual-branch architecture for high-resolution urban semantic segmentation that combines:

1. **XLET-NSST Feature Extraction**: Replaces traditional Haar DWT with Non-Subsampled Shearlet Transform (NSST), generating 87 frequency-domain channels from 3-channel RGB input
2. **Spatial Branch**: Processes raw RGB imagery with MambaVision encoder for global context
3. **XLET Branch**: Processes NSST features (87 channels) for localized edge and texture information
4. **Adaptive Fusion**: Multi-scale fusion modules with VMamba blocks for optimal feature integration
5. **Urban Context Decoder**: Dual attention mechanism (spatial 7×7 + channel) for sharp boundary preservation
6. **Composite Loss**: L_CE + L_Lovász for both semantic correctness and direct mIoU optimization

## Architecture Highlights

### NSST Feature Generation
- **Input**: RGB image [B, 3, H, W]
- **Output**: NSST features [B, 87, H, W]
- **Decomposition**: 3 scales with [2, 4, 8] directional bands = 29 subbands per channel × 3 channels = 87 total
- **Key Advantage**: Non-subsampled (maintains H×W resolution) for dense prediction tasks

### Dual-Branch Encoding
- **Spatial Branch**: MambaVision encoder with 3 input channels (RGB)
- **XLET Branch**: MambaVision encoder with 87 input channels (NSST features)
- **Critical Design**: Configurable `in_channels` parameter in stem layer accepts both 3 and 87 channels

### Fusion Strategy
At each encoder stage j:
```
F_j^fused = VMamba(Conv_1×1([F_j^spatial, F_j^xlet]))
```

### Urban Context Block (UCB)
```
A_spatial = σ(Conv_7×7([AvgPool(F), MaxPool(F)]))
A_channel = σ(MLP(GAP(F)) + MLP(GMP(F)))
F_enhanced = F ⊙ A_spatial ⊙ A_channel
```

## Directory Structure

```
urban_mamba/
├── README.md                # This file (project overview)
├── train.py                # Main training script
├── demo.py                 # Quick start demonstration
├── __init__.py             # Package initialization
│
├── models/                 # Architecture components
│   ├── __init__.py
│   ├── model.py            # Main UrbanMamba architecture
│   ├── mambavision_segmentation.py  # MambaVision encoder wrapper
│   ├── standalone_mambavision.py    # Standalone MambaVision implementation
│   ├── mamba_modules.py    # VSS blocks, encoders, fusion modules
│   ├── decoder.py          # Urban Context Decoder with UCB
│   ├── aggregation.py      # Multi-scale feature aggregation head
│   └── transforms.py       # NSST feature extraction (87 channels)
│
├── losses/                 # Loss functions and metrics
│   ├── __init__.py
│   ├── lovasz_loss.py     # Lovász-Softmax loss for mIoU optimization
│   └── metrics.py         # Evaluation metrics (mIoU, boundary F1)
│
├── configs/               # Configuration files
│   ├── config.yaml       # Training hyperparameters
│   └── requirements.txt  # Python dependencies
│
├── tests/                # Integration tests
│   ├── test_official_integration.py
│   └── verify_corrections.py
│
├── docs/                 # Documentation
│   ├── SYSTEM_ARCHITECTURE.md     # Comprehensive system guide
│   ├── PROJECT_STRUCTURE.md       # Reorganization details
│   ├── SYSTEM_CORRECTIONS.md      # File path fixes
│   ├── INTEGRATION_COMPLETE.md    # MambaVision integration
│   ├── INTEGRATION_STATUS.md      # Integration checklist
│   ├── MAMBAVISION_INTEGRATION.md # Integration guide
│   └── STRUCTURE_GUIDE.md         # Quick reference
│
├── data/                 # Dataset directory (create this)
│   └── urban_dataset/
│       ├── train/
│       └── val/
│
└── utils/                # Utilities
    └── download_dataset.py
```

**Note**: For comprehensive system documentation including architecture diagrams, pipeline flows, and development guides, see [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

## Installation

### Requirements
Install dependencies from the configuration folder:
```bash
pip install -r configs/requirements.txt
```

### Package Contents
All dependencies are listed in `configs/requirements.txt` including:
- PyTorch and torchvision
- NSST transform libraries (scipy, pywavelets)
- Training utilities (tensorboard, tqdm, pyyaml)
- MambaVision dependencies (timm, einops)

## Quick Start

### 1. Run Demo (Optional)

Test the complete pipeline with synthetic data:
```bash
python demo.py
```

This demonstrates:
- Model inference
- Training step with loss computation
- Dual-branch feature extraction
- Architecture inspection

### 2. Configuration

Edit `configs/config.yaml` to match your dataset and hardware:

```yaml
model:
  size: 'base'  # 'tiny', 'small', 'base', 'large'
  num_classes: 6

data:
  root: './data/urban_dataset'
  image_size: [512, 512]

training:
  epochs: 300
  batch_size: 8
  gradient_accumulation_steps: 2
```

### 3. Dataset Preparation

Organize your dataset as:
```
data/urban_dataset/
├── train/
│   ├── images/  # RGB images
│   └── labels/  # Segmentation masks
└── val/
    ├── images/
    └── labels/
```

Labels should be integer class indices (0-5 for 6 classes).

### 4. Training

```bash
# From the urban_mamba directory:
python train.py

# Or with custom config:
python train.py --config configs/custom_config.yaml
```

The training script will:
- Load configuration from `configs/config.yaml`
- Automatically compute NSST features on-the-fly
- Save checkpoints to `outputs/urban_mamba/`
- Log metrics to TensorBoard

### 5. Inference

```python
import torch
from models.model import UrbanMamba

# Load model
model = UrbanMamba(num_classes=6, model_size='base')
checkpoint = torch.load('outputs/urban_mamba/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
rgb_image = torch.randn(1, 3, 512, 512)  # Your RGB image

# Forward pass (NSST computed automatically)
with torch.no_grad():
    logits = model(rgb_image)
    predictions = torch.argmax(logits, dim=1)
```

**Import Pattern**: With the new organized structure, use:
```python
from models import UrbanMamba           # Main model
from losses import CompositeLoss        # Loss function
from losses import SegmentationMetrics  # Evaluation metrics
```

## Model Configurations

| Model Size | Hidden Dims        | Depths      | Parameters |
|------------|-------------------|-------------|------------|
| Tiny       | [64, 128, 256, 512]  | [2, 2, 4, 2]  | ~30M      |
| Small      | [96, 192, 384, 768]  | [2, 2, 6, 2]  | ~45M      |
| Base       | [96, 192, 384, 768]  | [2, 2, 8, 2]  | ~55M      |
| Large      | [128, 256, 512, 1024] | [2, 4, 12, 2] | ~95M      |

## Loss Function

### Composite Loss
```
L_total = λ₁ × L_CE + λ₂ × L_Lovász
```

**Default weights**:
- λ₁ = 1.0 (Cross-Entropy for semantic stability)
- λ₂ = 0.75 (Lovász-Softmax for boundary precision)

### Tuning Guidelines
- **Blurry boundaries**: Increase λ₂ (e.g., 1.0 or 1.5)
- **Training instability**: Increase λ₁, decrease λ₂
- **Class imbalance**: Add `class_weights` in config

## Optimization Strategy

### Learning Rate Schedule
1. **Warm-up**: Linear increase for 5 epochs (stable SSM initialization)
2. **Cosine Annealing**: Smooth decay to minimum LR

### Gradient Accumulation
Simulates larger batch sizes with limited VRAM:
```yaml
batch_size: 8
gradient_accumulation_steps: 2
# Effective batch size = 16
```

### Memory Management
- **Mixed Precision (AMP)**: Enable `use_amp: true` for 2× memory reduction
- **Gradient Checkpointing**: Add to model if needed for very deep models

## Evaluation Metrics

### Primary Metric: mIoU
Mean Intersection-over-Union across all classes

### Secondary Metrics
- **Pixel Accuracy**: Overall classification accuracy
- **Per-class IoU**: Individual class performance
- **Boundary F1**: Geometric accuracy of boundaries (critical for roads/buildings)

### Computing Metrics
```python
from losses.metrics import SegmentationMetrics

metrics = SegmentationMetrics(num_classes=6, class_names=['buildings', ...])
metrics.update(predictions, labels)
metrics.update_boundary(predictions, labels, threshold=2)

results = metrics.compute_all()
print(f"mIoU: {results['mIoU']:.4f}")
print(f"Boundary F1: {results['boundary_f1']:.4f}")
```

## Key Implementation Details

### 1. NSST Feature Extraction
- **On-the-fly generation**: Computed during training (no pre-storage needed)
- **Non-subsampled**: Maintains spatial resolution H×W
- **87 channels**: 3 RGB × 29 subbands per channel

### 2. Cross-Scan Mechanisms
- **Directional**: 4 diagonal passes for oriented features
- **Scale-aware**: Multi-stride for hierarchical patterns
- **Spiral**: Center-out for contextual dependencies

### 3. Fusion Module Architecture
```
Input: F_spatial [B, C, H, W], F_xlet [B, C, H, W]
Step 1: Concatenate → [B, 2C, H, W]
Step 2: Conv_1×1 → [B, C, H, W]
Step 3: VMamba block (adaptive integration)
Output: F_fused [B, C, H, W]
```

### 4. Urban Context Block (UCB)
- **7×7 spatial attention**: Large receptive field for thin features (roads)
- **Channel attention**: GAP + GMP for holistic + discriminative features
- **Dual modulation**: F ⊙ A_spatial ⊙ A_channel

## Performance Tips

### For High-Resolution Images (>512×512)
1. Use gradient accumulation to maintain effective batch size
2. Enable mixed precision training
3. Consider patch-based training if memory is limited

### For Class Imbalance
1. Set `class_weights` in config based on inverse frequency
2. Consider Focal Loss: `loss_type: 'focal'` in lovasz_loss.py
3. Use `classes: 'present'` in Lovász loss (default)

### For Sharp Boundaries
1. Increase λ_lovasz (e.g., 1.0 or higher)
2. Monitor boundary F1 metric during validation
3. Ensure NSST features are properly normalized

## Citation

If you use UrbanMamba in your research, please cite:

```bibtex
@article{urbanmamba2024,
  title={UrbanMamba: A Dual-Branch Architecture for High-Resolution Urban Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This implementation builds upon:
- **MambaVision**: Visual State Space Models
- **UrbanMamba Paper**: Dual-branch architecture for urban scenes
- **NSST**: Non-Subsampled Shearlet Transform for frequency analysis
- **Lovász-Softmax Loss**: Direct mIoU optimization

## License

MIT License - See LICENSE file for details

## Testing

Run integration tests to verify the complete pipeline:

```bash
# Test MambaVision integration
python tests/test_official_integration.py

# Verify all corrections
python tests/verify_corrections.py
```

These tests verify:
- Model instantiation and forward pass
- NSST feature extraction (87 channels)
- Dual-branch encoding with proper dimensions
- Fusion modules and decoder functionality
- Loss computation and gradient flow

## Development

### Adding New Features

1. **Models**: Add to `models/` directory
2. **Losses**: Add to `losses/` directory
3. **Configs**: Update `configs/config.yaml`
4. **Tests**: Add tests to `tests/` directory

### Code Organization

The project follows a modular structure:
- Root directory contains only user-facing scripts (`train.py`, `demo.py`)
- Implementation details are organized in subfolders
- All imports use relative package paths (e.g., `from models import ...`)
- Configuration is centralized in `configs/`

For detailed architecture documentation, see [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

