"""
UrbanMamba Models Package

v3.0: Original Twin-Tower architecture
v3.1: Production-ready with XLET Normalization Stem
"""

from .model import UrbanMambaV3, create_urban_mamba_v3
from .model_v31 import UrbanMambaV31, create_urbanmamba_v31
from .xlet_stem import XLETNormalizationStem, LightweightXLETStem
from .transforms import NSSTDecomposition
from .encoder import MambaVisionEncoder, create_mambavision_encoder
from .fusion import MambaFusionBlock
from .decoder import (
    UrbanContextDecoder,
    UrbanContextBlock,
    SpatialAttention,
    ChannelAttention,
    MultiScaleAggregationHead  # Alias for compatibility
)

__all__ = [
    # v3.0
    'UrbanMambaV3',
    'create_urban_mamba_v3',
    # v3.1 (Recommended)
    'UrbanMambaV31',
    'create_urbanmamba_v31',
    'XLETNormalizationStem',
    'LightweightXLETStem',
    # Components
    'NSSTDecomposition',
    'MambaVisionEncoder',
    'create_mambavision_encoder',
    'MambaFusionBlock',
    'UrbanContextDecoder',
    'UrbanContextBlock',
    'SpatialAttention',
    'ChannelAttention',
    'MultiScaleAggregationHead',
]
