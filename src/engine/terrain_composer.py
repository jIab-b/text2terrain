"""
Feature-based terrain generation engine.

Replaces the JAX-based TerrainEngine with simple, predictable
terrain generation using feature generators.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .feature_generators import (
    MountainGenerator, ValleyGenerator, CaveGenerator,
    RiverGenerator, BiomeGenerator, BaseGenerator
)
from .grid_manager import GridManager


class TerrainComposer:
    """
    Feature-based terrain generation with seamless grids.
    
    Generates predictable terrain features for LLM learning
    while supporting Minecraft-level complexity.
    """
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
        self.grid_manager = GridManager(tile_size)
        
        # Initialize feature generators
        self.feature_generators = {
            "base": BaseGenerator(),
            "mountains": MountainGenerator(),
            "valleys": ValleyGenerator(), 
            "caves": CaveGenerator(),
            "rivers": RiverGenerator(),
            "biomes": BiomeGenerator()
        }
        
        # Legacy module mapping for compatibility
        self.legacy_modules = {
            0: "perlin_noise",
            1: "ridged_multi", 
            2: "domain_warp",
            3: "hydraulic_erosion",
            4: "mountains",
            5: "valleys",
            6: "caves",
            7: "rivers", 
            8: "biomes"
        }
    
    def generate_heightmap(
        self,
        world_x: int,
        world_y: int, 
        module_ids: List[int] = None,
        parameters: Dict[str, float] = None,
        seeds: List[int] = None,
        global_seed: int = 42,
        features: Dict = None,
        legacy_mode: bool = True
    ) -> np.ndarray:
        """
        Generate heightmap using either legacy or new feature-based approach.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            module_ids: Legacy module IDs (for compatibility)
            parameters: Generation parameters
            seeds: Random seeds per module/feature
            global_seed: Global random seed
            features: New feature-based configuration
            legacy_mode: Whether to use legacy parameter-based generation
            
        Returns:
            Heightmap as numpy array of shape (tile_size, tile_size)
        """
        
        if legacy_mode and module_ids is not None:
            return self._generate_legacy(world_x, world_y, module_ids, parameters, seeds, global_seed)
        elif features is not None:
            return self._generate_feature_based(world_x, world_y, features, parameters, seeds, global_seed)
        else:
            # Default simple generation
            return self._generate_simple(world_x, world_y, parameters or {}, global_seed)
    
    def _generate_legacy(
        self,
        world_x: int, world_y: int,
        module_ids: List[int],
        parameters: Dict[str, float],
        seeds: List[int],
        global_seed: int
    ) -> np.ndarray:
        """Generate using legacy module-based approach for compatibility."""
        
        # Create coordinate grid
        x_coords = np.linspace(world_x, world_x + self.tile_size, self.tile_size, endpoint=False)
        y_coords = np.linspace(world_y, world_y + self.tile_size, self.tile_size, endpoint=False)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Initialize heightmap
        heightmap = np.zeros_like(X)
        
        # Apply modules in sequence (legacy behavior)
        for i, module_id in enumerate(module_ids):
            seed = seeds[i] if i < len(seeds) else global_seed + i
            
            if module_id == 0:  # perlin_noise
                noise = self._simple_perlin(X, Y, seed, parameters)
                heightmap += noise
            elif module_id == 1:  # ridged_multi
                ridged = self._ridged_noise(X, Y, seed, parameters)
                heightmap += ridged
            elif module_id == 2:  # domain_warp
                heightmap = self._domain_warp(heightmap, parameters, seed)
            elif module_id == 3:  # hydraulic_erosion
                heightmap = self._simple_erosion(heightmap, parameters, seed)
            elif module_id >= 4:  # New feature generators
                feature_name = self.legacy_modules.get(module_id, "base")
                if feature_name in self.feature_generators:
                    generator = self.feature_generators[feature_name]
                    heightmap = generator.apply(heightmap, X, Y, parameters, seed)
        
        # Normalize and scale
        heightmap = np.clip(heightmap, 0, None)
        height_scale = parameters.get("height_scale", 1000.0)
        heightmap = (heightmap / (heightmap.max() + 1e-6)) * height_scale
        
        return heightmap
    
    def _generate_feature_based(
        self,
        world_x: int, world_y: int,
        features: Dict,
        parameters: Dict[str, float],
        seeds: List[int],
        global_seed: int
    ) -> np.ndarray:
        """Generate using new feature-based approach."""
        
        # Create coordinate grid
        x_coords = np.linspace(world_x, world_x + self.tile_size, self.tile_size, endpoint=False)
        y_coords = np.linspace(world_y, world_y + self.tile_size, self.tile_size, endpoint=False)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Generate base terrain
        heightmap = self.feature_generators["base"].generate(X, Y, parameters, global_seed)
        
        # Apply primary features
        seed_idx = 0
        for feature_name in features.get("primary_features", []):
            if feature_name in self.feature_generators:
                seed = seeds[seed_idx] if seed_idx < len(seeds) else global_seed + seed_idx
                generator = self.feature_generators[feature_name]
                heightmap = generator.apply(heightmap, X, Y, parameters, seed)
                seed_idx += 1
        
        # Apply secondary features
        for feature_name in features.get("secondary_features", []):
            if feature_name in self.feature_generators:
                seed = seeds[seed_idx] if seed_idx < len(seeds) else global_seed + seed_idx
                generator = self.feature_generators[feature_name]
                heightmap = generator.enhance(heightmap, parameters, seed)
                seed_idx += 1
        
        # Apply grid continuity if specified
        if features.get("grid_continuity", {}).get("blend_edges", False):
            heightmap = self.grid_manager.apply_boundary_smoothing(
                heightmap, world_x, world_y, features["grid_continuity"]
            )
        
        # Normalize and scale (same as legacy method)
        heightmap = np.clip(heightmap, 0, None)
        height_scale = parameters.get("height_scale", 1000.0)
        if heightmap.max() > 1e-6:
            heightmap = (heightmap / heightmap.max()) * height_scale
        
        return heightmap
    
    def _generate_simple(
        self,
        world_x: int, world_y: int,
        parameters: Dict[str, float],
        global_seed: int
    ) -> np.ndarray:
        """Simple fallback generation."""
        
        x_coords = np.linspace(world_x, world_x + self.tile_size, self.tile_size, endpoint=False)
        y_coords = np.linspace(world_y, world_y + self.tile_size, self.tile_size, endpoint=False)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Simple Perlin noise
        heightmap = self._simple_perlin(X, Y, global_seed, parameters)
        height_scale = parameters.get("height_scale", 1000.0)
        
        return heightmap * height_scale
    
    def _simple_perlin(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """Simple Perlin noise implementation (replaces JAX version)."""
        
        frequency = parameters.get("frequency", 0.01)
        octaves = int(parameters.get("octaves", 4))
        persistence = parameters.get("persistence", 0.5)
        lacunarity = parameters.get("lacunarity", 2.0)
        
        # Scale coordinates
        x = X * frequency
        y = Y * frequency
        
        total = np.zeros_like(x)
        amplitude = 1.0
        max_value = 0.0
        
        np.random.seed(seed)
        
        for i in range(octaves):
            # Simple noise using numpy
            noise = self._noise2d(x, y, seed + i)
            total += noise * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            x *= lacunarity
            y *= lacunarity
        
        return total / max_value
    
    def _noise2d(self, x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        """Simple 2D noise function."""
        
        # Grid coordinates
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional parts
        fx = x - x0
        fy = y - y0
        
        # Smooth interpolation
        u = fx * fx * (3 - 2 * fx)
        v = fy * fy * (3 - 2 * fy)
        
        # Hash function for random values
        def hash2d(ix, iy):
            h = (ix * 374761393 + iy * 668265263 + seed * 1664525) % 2147483647
            return (h / 2147483647.0) * 2.0 - 1.0
        
        # Get corner values
        c00 = hash2d(x0, y0)
        c10 = hash2d(x1, y0) 
        c01 = hash2d(x0, y1)
        c11 = hash2d(x1, y1)
        
        # Bilinear interpolation
        top = c00 + u * (c10 - c00)
        bottom = c01 + u * (c11 - c01)
        
        return top + v * (bottom - top)
    
    def _ridged_noise(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """Ridged multifractal noise for mountain ridges."""
        
        base_noise = self._simple_perlin(X, Y, seed, parameters)
        ridge_sharpness = parameters.get("ridge_sharpness", 1.0)
        
        # Create ridges by inverting absolute values
        ridged = 1.0 - np.abs(base_noise)
        ridged = np.power(ridged, ridge_sharpness)
        
        return ridged
    
    def _domain_warp(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float], 
        seed: int
    ) -> np.ndarray:
        """Simple domain warping."""
        
        warp_amplitude = parameters.get("warp_amplitude", 100.0)
        
        if warp_amplitude == 0:
            return heightmap
        
        # Use heightmap values to create displacement
        rows, cols = heightmap.shape
        y_indices, x_indices = np.ogrid[:rows, :cols]
        
        # Simple displacement based on heightmap values
        dx = (heightmap - 0.5) * warp_amplitude * 0.01
        dy = (heightmap - 0.5) * warp_amplitude * 0.01
        
        # Apply displacement with bounds checking
        new_x = np.clip(x_indices + dx, 0, cols - 1).astype(int)
        new_y = np.clip(y_indices + dy, 0, rows - 1).astype(int)
        
        return heightmap[new_y, new_x]
    
    def _simple_erosion(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Simple hydraulic erosion simulation."""
        
        erosion_speed = parameters.get("erosion_speed", 0.1)
        iterations = int(parameters.get("iterations", 50))
        
        if erosion_speed == 0 or iterations == 0:
            return heightmap
        
        result = heightmap.copy()
        
        # Simple erosion: smooth high areas, deepen low areas
        for _ in range(iterations):
            # Calculate gradients
            grad_x = np.gradient(result, axis=1)
            grad_y = np.gradient(result, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Erode steep areas
            erosion_mask = gradient_magnitude > np.percentile(gradient_magnitude, 70)
            result[erosion_mask] -= erosion_speed * 0.1
            
            # Deposit in flat areas
            deposition_mask = gradient_magnitude < np.percentile(gradient_magnitude, 30)
            result[deposition_mask] += erosion_speed * 0.05
        
        return result
    
    def generate_seamless_region(
        self,
        center_x: int, center_y: int,
        grid_size: int = 3,
        **kwargs
    ) -> Dict[tuple, np.ndarray]:
        """Generate seamless grid of terrain tiles."""
        
        return self.grid_manager.generate_seamless_region(
            center_x, center_y, grid_size, self, **kwargs
        )