"""
Feature-based terrain generation engine.

This module provides terrain generation functionality with:
- Feature-based terrain composition (mountains, valleys, caves, etc.)
- Seamless grid generation for infinite worlds
- Terrain analysis and validation
- Legacy compatibility with existing renderer
"""

from ..engine import TerrainComposer, GridManager, HeightmapAnalyzer
from .grammar import ModuleRegistry, ParameterSpec

# Backward compatibility alias
TerrainEngine = TerrainComposer

__all__ = [
    "TerrainComposer",
    "TerrainEngine",  # Legacy alias
    "GridManager",
    "HeightmapAnalyzer", 
    "ModuleRegistry", 
    "ParameterSpec"
]