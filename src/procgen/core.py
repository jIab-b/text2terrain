"""
Core terrain generation engine using JAX for differentiable operations.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple
from .grammar import ModuleRegistry, ParameterSpec
from .modules import noise, warp, erosion
from .jax_backend import generate as fast_generate


class TerrainEngine:
    """
    Main terrain generation engine that orchestrates modules based on parameters.
    
    This engine:
    - Takes world coordinates and generates heightmaps
    - Supports differentiable operations via JAX
    - Manages module composition and parameter validation
    """
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
        self.registry = ModuleRegistry()
        
        # Register built-in modules
        self._register_builtin_modules()
    
    def _register_builtin_modules(self):
        """Register all built-in terrain modules."""
        
        # Noise modules
        self.registry.register(
            "perlin_noise",
            noise.perlin_noise,
            ParameterSpec({
                "frequency": (0.001, 0.1, 0.01),
                "octaves": (1, 8, 4),
                "persistence": (0.1, 0.9, 0.5),
                "lacunarity": (1.5, 3.0, 2.0)
            })
        )
        
        self.registry.register(
            "ridged_multi",
            noise.ridged_multifractal,
            ParameterSpec({
                "frequency": (0.001, 0.1, 0.01),
                "octaves": (1, 8, 4),
                "persistence": (0.1, 0.9, 0.5),
                "ridge_sharpness": (0.1, 2.0, 0.9)
            })
        )
        
        # Domain warping
        self.registry.register(
            "domain_warp",
            warp.domain_warp,
            ParameterSpec({
                "warp_amplitude": (10.0, 500.0, 100.0),
                "warp_frequency": (0.001, 0.01, 0.005),
                "warp_octaves": (1, 4, 2)
            })
        )
        
        # Erosion
        self.registry.register(
            "hydraulic_erosion",
            erosion.hydraulic_erosion,
            ParameterSpec({
                "iterations": (10, 1000, 200),
                "rain_amount": (0.1, 2.0, 0.5),
                "evaporation": (0.01, 0.1, 0.05),
                "capacity": (0.1, 1.0, 0.3)
            })
        )
    
    def generate_tile(
        self,
        world_x: int,
        world_y: int,
        module_ids: List[int],
        parameters: Dict[str, float],
        seeds: List[int],
        global_seed: int = 42,
        use_fast: bool = True
    ) -> jnp.ndarray:
        """
        Generate a terrain tile at world coordinates.
        
        Args:
            world_x: World X coordinate (integer multiple of tile_size)
            world_y: World Y coordinate (integer multiple of tile_size)
            module_ids: List of module IDs to apply
            parameters: Dictionary of parameter values
            seeds: Per-module random seeds
            global_seed: Global random seed
            
        Returns:
            jnp.ndarray: Height map of shape (tile_size, tile_size)
        """
        
        # Fast deterministic path
        if use_fast and "height_scale" in parameters:
            p = dict(parameters)
            p.setdefault("seed", int(global_seed & 0xFFFFFFFF))
            return fast_generate(p)
        
        # Legacy module chain
        x_coords = jnp.linspace(
            world_x, world_x + self.tile_size,
            self.tile_size, endpoint=False
        )
        y_coords = jnp.linspace(
            world_y, world_y + self.tile_size, 
            self.tile_size, endpoint=False
        )
        
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Initialize height map
        heightmap = jnp.zeros_like(X)
        
        # Apply each module in sequence
        for i, module_id in enumerate(module_ids):
            module_name = self.registry.get_module_name(module_id)
            module_func = self.registry.get_module_function(module_id)
            param_spec = self.registry.get_parameter_spec(module_id)
            
            # Extract parameters for this module
            module_params = param_spec.extract_params(parameters)
            
            # Use per-module seed
            seed = seeds[i] if i < len(seeds) else global_seed + i
            
            # Apply module
            if module_name in ["perlin_noise", "ridged_multi"]:
                # Noise modules generate new terrain
                noise_result = module_func(X, Y, seed=seed, **module_params)
                heightmap = heightmap + noise_result
                
            elif module_name == "domain_warp":
                # Warp modules modify coordinates
                X_warped, Y_warped = module_func(X, Y, seed=seed, **module_params)
                X, Y = X_warped, Y_warped
                
            elif module_name == "hydraulic_erosion":
                # Erosion modules modify existing heightmap
                heightmap = module_func(heightmap, seed=seed, **module_params)
        
        return heightmap
    
    def generate_with_neighbors(
        self,
        world_x: int,
        world_y: int,
        module_ids: List[int],
        parameters: Dict[str, float],
        seeds: List[int],
        global_seed: int = 42
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Generate a tile and its four neighbors for edge blending.
        
        Returns:
            Tuple of (center_tile, neighbors_dict)
        """
        
        center = self.generate_tile(world_x, world_y, module_ids, parameters, seeds, global_seed)
        
        neighbors = {}
        offsets = {
            'north': (0, -self.tile_size),
            'south': (0, self.tile_size),
            'east': (self.tile_size, 0),
            'west': (-self.tile_size, 0)
        }
        
        for direction, (dx, dy) in offsets.items():
            neighbors[direction] = self.generate_tile(
                world_x + dx, world_y + dy,
                module_ids, parameters, seeds, global_seed
            )
        
        return center, neighbors
    
    def validate_parameters(self, module_ids: List[int], parameters: Dict[str, float]) -> bool:
        """Validate that all required parameters are present and in valid ranges."""
        
        for module_id in module_ids:
            param_spec = self.registry.get_parameter_spec(module_id)
            if not param_spec.validate(parameters):
                return False
        
        return True