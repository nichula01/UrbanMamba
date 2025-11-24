# UrbanMamba: Dual-Branch Architecture with XLET-NSST Feature Swap

High-resolution urban semantic segmentation using dual-branch architecture with NSST frequency decomposition.

## Quick Start

```bash
cd semantic_segmentation/urban_mamba
pip install -r requirements.txt
python test_implementation.py  # Verify installation
```

## Repository Structure

```
MambaVision-main/
├── LICENSE                                    # MIT License
├── README.md                                  # This file
├── setup.py                                   # Package setup
├── setup.cfg                                  # Setup configuration
├── mambavision/                              # MambaVision backbone (NVlabs)
│   ├── models/                               # Pre-trained models
│   ├── mamba_vision.py                       # MambaVision architecture
│   └── ...                                   # Other utilities
└── semantic_segmentation/
    └── urban_mamba/                          # UrbanMamba implementation
        ├── __init__.py                       # Package initialization
        ├── aggregation.py                    # Multi-scale aggregation head
        ├── config.yaml                       # Training configuration
        ├── decoder.py                        # Urban Context Decoder
        ├── lovasz_loss.py                    # Lovász-Softmax loss
        ├── mamba_modules.py                  # VSS blocks, encoders, fusion
        ├── metrics.py                        # Evaluation metrics
        ├── model.py                          # Main UrbanMamba model
        ├── README.md                         # Detailed documentation
        ├── requirements.txt                  # Python dependencies
        ├── test_implementation.py            # Test suite (8/8 passing)
        ├── train.py                          # Training pipeline
        └── transforms.py                     # NSST decomposition
```

## Features

### MambaVision Backbone (from NVlabs)
- **Pre-trained Models**: Efficient Vision Mamba architecture
- **State Space Models**: Linear complexity for high-resolution images
- **ImageNet Weights**: Available for transfer learning

### UrbanMamba Extension
- **NSST Transform**: Generates 87 frequency channels from 3-channel RGB
- **Dual-Branch Encoder**: Spatial (3ch) + XLET (87ch) paths using MambaVision
- **VSS Blocks**: Visual State Space with directional scanning
- **Urban Context Decoder**: 7×7 spatial + channel attention
- **Advanced Loss**: Lovász-Softmax for direct mIoU optimization
- **37M Parameters**: Efficient architecture for high-resolution images

## Usage

```python
import torch
from semantic_segmentation.urban_mamba.model import UrbanMamba

# Create model
model = UrbanMamba(num_classes=6)

# Forward pass
image = torch.randn(1, 3, 512, 512)
output = model(image)  # [1, 6, 512, 512]
```

## Testing

All 8 tests passing (100%):
- ✓ NSST Decomposition (87 channels)
- ✓ MambaVision Encoders (3ch + 87ch)
- ✓ Fusion Module
- ✓ Urban Context Decoder
- ✓ Complete Model (37M params)
- ✓ Loss Functions
- ✓ Multi-Scale Aggregation
- ✓ Gradient Flow

## Training

```bash
cd semantic_segmentation/urban_mamba
python train.py --config config.yaml --data_root /path/to/dataset
```

## Documentation

See `semantic_segmentation/urban_mamba/README.md` for detailed:
- Architecture overview
- Module descriptions
- Training guide
- API reference

## Citation

```bibtex
@software{urbanmamba2024,
  title={UrbanMamba: Dual-Branch Architecture with XLET-NSST Feature Swap},
  author={Urban Segmentation Team},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details
