"""
Enhanced data generation pipeline for Text2Terrain.

Generates high-quality training data by:
1. Sampling feature-based terrain configurations
2. Generating actual heightmaps using terrain composer
3. Analyzing terrain features for validation
4. Creating natural language captions from analysis
5. Validating terrain-caption consistency
6. Saving as enhanced JSON with legacy compatibility
"""

from .generator_v2 import DatasetGeneratorV2, main
from .feature_captions import FeatureCaptionGenerator
from .terrain_validator import TerrainValidator
from .quality_metrics import QualityMetrics
from .preprocessing import preprocess_dataset

# Backward compatibility aliases
DatasetGenerator = DatasetGeneratorV2
CaptionGenerator = FeatureCaptionGenerator

__all__ = [
    "DatasetGeneratorV2", "DatasetGenerator",  # New and legacy names
    "FeatureCaptionGenerator", "CaptionGenerator",  # New and legacy names
    "TerrainValidator", "QualityMetrics",
    "preprocess_dataset", "main"
]