"""
Data generation pipeline for Text2Terrain.

Generates synthetic training data by:
1. Sampling random terrain parameters
2. Generating heightmaps using procgen engine
3. Creating natural language captions
4. Saving as structured JSON traces
"""

from .generator import DatasetGenerator, main
from .captions import CaptionGenerator
from .preprocessing import preprocess_dataset

__all__ = ["DatasetGenerator", "CaptionGenerator", "preprocess_dataset", "main"]