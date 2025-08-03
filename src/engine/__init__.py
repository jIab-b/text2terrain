"""
New feature-based terrain generation engine.

Replaces JAX-based procgen with simple, predictable terrain generation
focused on LLM learnability and Minecraft-level terrain variety.
"""

from .terrain_composer import TerrainComposer
from .grid_manager import GridManager
from .heightmap_analyzer import HeightmapAnalyzer

__all__ = ["TerrainComposer", "GridManager", "HeightmapAnalyzer"]