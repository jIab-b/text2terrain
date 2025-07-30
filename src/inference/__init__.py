"""
Inference components for Text2Terrain.

Lightweight deployment-ready modules for text-to-terrain generation:
- Text-to-parameter prediction
- FastAPI server for model serving
- Terrain sampling interface
"""

from .text2param import Text2ParamPredictor
from .sampler import TerrainSampler
from .api import create_app, main

__all__ = ["Text2ParamPredictor", "TerrainSampler", "create_app", "main"]