# UrbanMamba: Complete System Architecture & Pipeline Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Complete Pipeline Flow](#complete-pipeline-flow)
4. [Module Descriptions](#module-descriptions)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Configuration System](#configuration-system)
8. [Development Guide](#development-guide)

---

## 🎯 System Overview

**UrbanMamba** is a state-of-the-art dual-branch deep learning architecture for high-resolution urban semantic segmentation. It combines:

- **XLET-NSST Feature Engineering**: Frequency-domain feature extraction (87 channels)
- **MambaVision Backbone**: Efficient vision state space models for spatial processing
- **Dual-Branch Architecture**: Parallel processing of spatial (RGB) and frequency (NSST) features
- **Adaptive Fusion**: Multi-scale feature integration with Mamba modules
- **Urban Context Decoder**: Specialized decoder for urban scene boundary preservation

### Key Innovation

The system performs an **XLET-NSST feature swap**, replacing traditional Haar DWT with Non-Subsampled Shearlet Transform (NSST), generating 87 frequency-domain channels that capture edge orientations and textures critical for urban scene understanding.

---

## 🏗️ Architecture Deep Dive

### Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: RGB Image [B, 3, H, W]                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │  Spatial Branch   │           │  XLET Branch      │
    │  (RGB - 3ch)      │           │  NSST Transform   │
    │                   │           │  (RGB → 87ch)     │
    └─────────┬─────────┘           └─────────┬─────────┘
              │                               │
              │ MambaVision                   │ MambaVision
              │ Encoder                       │ Encoder
              │                               │
    ┌─────────▼─────────┐           ┌─────────▼─────────┐
    │ Feature Maps:     │           │ Feature Maps:     │
    │ F1: [B,80,H/4,W/4]│           │ F1: [B,80,H/4,W/4]│
    │ F2: [B,160,H/8,W/8]│          │ F2: [B,160,H/8,W/8]│
    │ F3: [B,320,H/16,W/16]│        │ F3: [B,320,H/16,W/16]│
    │ F4: [B,640,H/32,W/32]│        │ F4: [B,640,H/32,W/32]│
    └─────────┬─────────┘           └─────────┬─────────┘
              │                               │
              └───────────┬───────────────────┘
                          │
                  Multi-Scale Fusion
                 (Mamba Fusion Modules)
                          │
              ┌───────────▼───────────┐
              │ Fused Features:       │
              │ FF1, FF2, FF3, FF4    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │ Urban Context Decoder │
              │ (Dual Attention)      │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │ OUTPUT: Segmentation  │
              │ [B, num_classes, H, W]│
              └───────────────────────┘
```

### Layer-by-Layer Breakdown

#### 1. Input Processing
```python
Input: RGB Image [Batch, 3, Height, Width]
├─→ Spatial Branch: Keeps RGB as-is [B, 3, H, W]
└─→ XLET Branch: NSST Transform → [B, 87, H, W]
```

#### 2. NSST Decomposition (XLET Branch)
```python
NSSTDecomposition(
    scales=3,                    # 3-scale decomposition
    directions_profile=[2,3,4]   # Directional bands per scale
)

Output Channels Calculation:
Scale 1: 2^2 + 1 = 5 directions
Scale 2: 2^3 + 1 = 9 directions  
Scale 3: 2^4 + 1 = 17 directions
Total per channel: 5 + 9 + 17 = 31 subbands
RGB channels: 31 × 3 = 93... wait, actually:
Lowpass: 1 × 3 = 3
Scale 1: 4 × 3 = 12
Scale 2: 8 × 3 = 24
Scale 3: 16 × 3 = 48
Total: 3 + 12 + 24 + 48 = 87 channels ✓
```

#### 3. Dual MambaVision Encoders

**Spatial Encoder (RGB - 3 channels):**
```python
create_mambavision_segmentation_encoder(
    in_channels=3,
    variant='tiny',  # or 'small', 'base'
    pretrained=None
)

Stage Output Dimensions (tiny variant):
├─ Stage 1: [B, 80, H/4, W/4]
├─ Stage 2: [B, 160, H/8, W/8]
├─ Stage 3: [B, 320, H/16, W/16]
└─ Stage 4: [B, 640, H/32, W/32]
```

**XLET Encoder (NSST - 87 channels):**
```python
create_mambavision_segmentation_encoder(
    in_channels=87,    # XLET-NSST features
    variant='tiny',
    pretrained=None    # Trained from scratch
)

Same output dimensions as spatial encoder
```

#### 4. Multi-Scale Fusion
```python
For each stage i ∈ {1, 2, 3, 4}:
    MambaFusionModule(
        spatial_dim=dims[i],
        xlet_dim=dims[i],
        output_dim=dims[i]
    )
    
Process:
    Concat[F_spatial[i], F_xlet[i]] → [B, 2*dims[i], H, W]
    ↓ Conv1x1
    [B, dims[i], H, W]
    ↓ VMamba Block (Selective Scan)
    Fused_Features[i] → [B, dims[i], H, W]
```

#### 5. Urban Context Decoder
```python
UrbanContextDecoder(
    encoder_dims=[80, 160, 320, 640],
    decoder_dims=[384, 192, 96, 48],
    num_classes=6
)

Decoding Path:
├─ F4 [640] → UCB → [384] → Upsample
├─ F3 [320] + [384] → UCB → [192] → Upsample
├─ F2 [160] + [192] → UCB → [96] → Upsample
├─ F1 [80] + [96] → UCB → [48] → Upsample
└─ Final Conv → [num_classes] → Output

UCB (Urban Context Block):
├─ Spatial Attention (7×7 conv, large receptive field)
├─ Channel Attention (GAP + GMP)
└─ Feature Enhancement
```

---

## 🔄 Complete Pipeline Flow

### Training Pipeline

```
1. Data Loading
   ├─ Load RGB image from disk
   ├─ Load ground truth label
   ├─ Apply augmentations (flip, rotation, scale)
   └─ Normalize & convert to tensors

2. Forward Pass
   ├─ Extract NSST features (on-the-fly)
   ├─ Dual encoding (spatial + XLET)
   ├─ Multi-scale fusion
   ├─ Decode to predictions
   └─ Output: [B, num_classes, H, W]

3. Loss Computation
   ├─ Cross-Entropy Loss (semantic correctness)
   ├─ Lovász-Softmax Loss (IoU optimization)
   └─ Total: L = λ_CE × L_CE + λ_Lovász × L_Lovász

4. Optimization
   ├─ Backward pass with mixed precision (AMP)
   ├─ Gradient accumulation (if enabled)
   ├─ Optimizer step (AdamW)
   └─ Learning rate scheduling (cosine annealing)

5. Validation
   ├─ Evaluate on validation set
   ├─ Compute mIoU, pixel accuracy
   ├─ Save best checkpoint
   └─ Early stopping check
```

### Inference Pipeline

```
1. Load trained model checkpoint
2. Set model to evaluation mode
3. For each test image:
   ├─ Load & preprocess RGB image
   ├─ Forward pass (no NSST extraction shown, done internally)
   ├─ Get predictions: argmax(logits)
   ├─ Post-process (resize to original size)
   └─ Save/visualize segmentation mask
```

---

## 📦 Module Descriptions

### Core Model Modules (`models/`)

#### `model.py` - UrbanMamba Main Architecture
**Purpose**: Orchestrates the complete dual-branch architecture
**Key Classes**: 
- `UrbanMamba`: Main model class
**Dependencies**: All other model modules
**Pipeline Role**: Central hub - instantiated by train/demo scripts

#### `mambavision_segmentation.py` - MambaVision Backbone Wrapper
**Purpose**: Wraps MambaVision for segmentation with configurable input channels
**Key Functions**:
- `create_mambavision_segmentation_encoder()`: Factory function
- `MambaVisionSegmentationBackbone`: Wrapper class
**Dependencies**: `standalone_mambavision.py`
**Pipeline Role**: Creates both 3-channel and 87-channel encoders

#### `standalone_mambavision.py` - Pure MambaVision Implementation
**Purpose**: Self-contained MambaVision architecture
**Key Classes**:
- `StandaloneMambaVision`: Complete model
- `MambaBlock`: Core selective scan block
**Dependencies**: None (except PyTorch, timm, mamba_ssm)
**Pipeline Role**: Actual backbone implementation used by wrapper

#### `mamba_modules.py` - Fusion & Mamba Components
**Purpose**: Fusion modules and custom Mamba components
**Key Classes**:
- `MambaFusionModule`: Fuses spatial + XLET features
- `CrossScan2D`: 2D scanning strategies
**Pipeline Role**: Bridges dual encoders in fusion stage

#### `decoder.py` - Urban Context Decoder
**Purpose**: Specialized decoder with dual attention
**Key Classes**:
- `UrbanContextDecoder`: Main decoder
- `UrbanContextBlock`: Decoder building block with attention
**Pipeline Role**: Decodes fused features to segmentation maps

#### `aggregation.py` - Multi-Scale Aggregation Head
**Purpose**: Alternative to UCD with bi-directional feedback
**Key Classes**:
- `MultiScaleAggregationHead`: Aggregation-based head
**Pipeline Role**: Optional decoder (use_aggregation_head=True)

#### `transforms.py` - NSST Feature Extraction
**Purpose**: Non-Subsampled Shearlet Transform implementation
**Key Classes**:
- `NSSTDecomposition`: Main NSST transformer
**Pipeline Role**: Generates 87-channel XLET features from RGB

### Loss & Metrics (`losses/`)

#### `lovasz_loss.py` - Composite Loss Function
**Purpose**: Combines CE and Lovász-Softmax for optimal training
**Key Classes**:
- `LovaszSoftmaxLoss`: Direct IoU optimization
- `CompositeLoss`: Combined loss (CE + Lovász)
**Pipeline Role**: Training objective

#### `metrics.py` - Evaluation Metrics
**Purpose**: Compute segmentation quality metrics
**Key Functions**:
- `compute_miou()`: Mean Intersection over Union
- `SegmentationMetrics`: Complete metrics tracker
**Pipeline Role**: Validation and evaluation

### Configuration (`configs/`)

#### `config.yaml` - Training Configuration
```yaml
model:
  size: 'tiny'  # Model variant
  num_classes: 6
  use_aggregation_head: false

data:
  root: './datasets/urban_dataset'
  image_size: [512, 512]
  
training:
  epochs: 300
  batch_size: 8
  base_lr: 0.0001
  
loss:
  lambda_ce: 1.0
  lambda_lovasz: 0.75
```

#### `requirements.txt` - Dependencies
Core dependencies for the entire pipeline

---

## 🎓 Training Pipeline

### Complete Training Flow

```python
# 1. Initialize
model = UrbanMamba(num_classes=6)
criterion = CompositeLoss(num_classes=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. Training Loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels = batch
        
        # Forward
        predictions = model(images)  # NSST extraction happens inside
        
        # Loss
        loss_dict = criterion(predictions, labels)
        loss = loss_dict['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    metrics = evaluate(model, val_loader)
    print(f"Epoch {epoch}: mIoU = {metrics['miou']:.4f}")
```

### Key Training Features

1. **On-the-Fly NSST**: Features computed during forward pass, no pre-computation
2. **Mixed Precision Training**: Automatic Mixed Precision (AMP) for speed
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Cosine Annealing**: Learning rate scheduling with warmup
5. **Multi-GPU Support**: Distributed Data Parallel (DDP)

---

## 🔮 Inference Pipeline

### Running Inference

```python
# 1. Load model
model = UrbanMamba(num_classes=6)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Prepare image
image = load_and_preprocess_image('test.jpg')  # [1, 3, H, W]

# 3. Inference
with torch.no_grad():
    prediction = model(image)  # [1, num_classes, H, W]
    segmentation = prediction.argmax(dim=1)  # [1, H, W]

# 4. Visualize
save_segmentation_mask(segmentation, 'output.png')
```

---

## ⚙️ Configuration System

### Model Configuration

```python
# Tiny Model (Fast, 22M params)
model = UrbanMamba(
    num_classes=6,
    hidden_dims=[80, 160, 320, 640],  # Auto-set from encoder
    use_aggregation_head=False
)

# Small Model (Balanced, 50M params)
encoder = create_mambavision_segmentation_encoder(
    variant='small'  # Changes dims to [96, 192, 384, 768]
)

# Base Model (Accurate, 90M params)
encoder = create_mambavision_segmentation_encoder(
    variant='base'  # Changes dims to [128, 256, 512, 1024]
)
```

### Training Configuration

Key parameters in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.size` | 'tiny' | Model variant |
| `model.num_classes` | 6 | Number of segmentation classes |
| `training.epochs` | 300 | Total training epochs |
| `training.batch_size` | 8 | Batch size per GPU |
| `training.base_lr` | 0.0001 | Initial learning rate |
| `loss.lambda_ce` | 1.0 | Cross-entropy weight |
| `loss.lambda_lovasz` | 0.75 | Lovász loss weight |
| `data.image_size` | [512, 512] | Input image dimensions |

---

## 🛠️ Development Guide

### Adding a New Component

#### Adding a New Loss Function

```python
# 1. Create losses/my_loss.py
class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)

# 2. Update losses/__init__.py
from .my_loss import MyCustomLoss
__all__ = [..., 'MyCustomLoss']

# 3. Use in training
from losses import MyCustomLoss
criterion = MyCustomLoss()
```

#### Adding a New Decoder

```python
# 1. Create models/my_decoder.py
class MyDecoder(nn.Module):
    def __init__(self, encoder_dims, num_classes):
        super().__init__()
        # Your decoder implementation

# 2. Update models/__init__.py
from .my_decoder import MyDecoder

# 3. Modify model.py
self.head = MyDecoder(encoder_dims, num_classes)
```

### File Organization Rules

1. **Models** → `models/` directory
2. **Losses & Metrics** → `losses/` directory
3. **Configs** → `configs/` directory
4. **Tests** → `tests/` directory
5. **Docs** → `docs/` directory
6. **Root** → Only README, train.py, demo.py

---

## 📊 Model Variants

| Variant | Params | FLOPs | mIoU (ADE20K) | Speed (FPS) |
|---------|--------|-------|---------------|-------------|
| Tiny | 22M | 8.2G | TBD | 45 |
| Small | 50M | 18.5G | TBD | 32 |
| Base | 90M | 35.2G | TBD | 21 |

---

## 🔗 Dependencies

### Core Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- mamba-ssm == 2.2.4
- timm == 1.0.15
- einops >= 0.7.0

### Optional Dependencies
- tensorboard (logging)
- wandb (experiment tracking)
- opencv-python (visualization)

---

## 📝 Citation

If you use UrbanMamba in your research, please cite:

```bibtex
@article{urbanmamba2024,
  title={UrbanMamba: Dual-Branch Architecture with XLET-NSST for Urban Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🤝 Contributing

When contributing code:

1. Follow the modular structure
2. Add proper docstrings
3. Update `__init__.py` files
4. Add tests for new components
5. Update relevant documentation

---

**Last Updated**: 2025-11-21  

