"""
Procedural terrain generation engine using JAX for differentiable operations.

This module provides the core terrain generation functionality with:
- Differentiable noise functions (Perlin, OpenSimplex, Worley)
- Domain warping operations
- Hydraulic erosion simulation
- Module registry and parameter management
"""

from .core import TerrainEngine
from .grammar import ModuleRegistry, ParameterSpec
from .modules import noise, warp, erosion
from .jax_backend import generate as jax_generate

__all__ = [
    "TerrainEngine",
    "ModuleRegistry", 
    "ParameterSpec",
    "noise",
    "warp", 
    "erosion",
    "jax_generate"
]