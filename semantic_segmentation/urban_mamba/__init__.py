"""
UrbanMamba v3: Twin Tower Architecture for Urban Semantic Segmentation
"""

__version__ = '3.0.0'
__author__ = 'UrbanMamba Team'

from .models import (
    UrbanMambaV3,
    create_urban_mamba_v3,
    NSSTDecomposition,
    MambaVisionEncoder,
    MambaFusionBlock,
    MultiScaleAggregationHead
)

from .losses import CompositeLoss, SegmentationMetrics

__all__ = [
    'UrbanMambaV3',
    'create_urban_mamba_v3',
    'NSSTDecomposition',
    'MambaVisionEncoder',
    'MambaFusionBlock',
    'MultiScaleAggregationHead',
    'CompositeLoss',
    'SegmentationMetrics',
]
