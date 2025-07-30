"""
Training components for Text2Terrain LoRA model.

Fireworks AI compatible training pipeline with:
- LoRA fine-tuning of Mistral-7B
- Multi-task loss (module classification + parameter regression)
- Efficient data loading and preprocessing
"""

from .model import Text2TerrainModel
from .datamodule import TerrainDataModule
from .train import main

__all__ = ["Text2TerrainModel", "TerrainDataModule", "main"]