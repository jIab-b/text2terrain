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
            return self._generate_legacy(world_x, world_y, module_ids, parameters, seeds, global_seed, features)
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
        global_seed: int,
        features: Dict = None
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
        
        # Realistic terrain-type-aware scaling
        heightmap = np.clip(heightmap, 0, None)
        
        # Get terrain type from features
        terrain_type = "plains"  # default
        if features and "terrain_type" in features:
            terrain_type = features["terrain_type"]
        
        # Define realistic height scales (in meters)
        terrain_scales = {
            "plains": 50,      # 0-50m elevation
            "hills": 300,      # 0-300m elevation  
            "valleys": 200,    # 0-200m with valley depth
            "mountains": 1200  # 0-1200m elevation
        }
        
        # Base elevations (sea level reference)
        base_elevations = {
            "plains": 5,       # slightly above sea level
            "hills": 20,       # rolling hills start higher
            "valleys": 0,      # valleys can be at sea level
            "mountains": 100   # mountains start on elevated terrain
        }
        
        # Get appropriate scaling
        max_height = terrain_scales.get(terrain_type, 200)
        base_elevation = base_elevations.get(terrain_type, 10)
        
        # Frequency-based height adjustment 
        frequency = parameters.get("frequency", 0.01)
        if frequency > 0.02:  # High frequency = small features = lower heights
            max_height *= 0.6
        elif frequency < 0.005:  # Low frequency = large features = can be taller
            max_height *= 1.4
        
        # Complexity-based adjustment
        if features and "complexity" in features:
            complexity = features["complexity"]
            if complexity > 0.7:  # High complexity = more dramatic terrain
                max_height *= 1.3
            elif complexity < 0.3:  # Low complexity = gentler terrain
                max_height *= 0.7
        
        # Apply realistic scaling
        if heightmap.max() > 1e-6:
            # Normalize to 0-1, then scale to realistic heights
            normalized = heightmap / heightmap.max()
            heightmap = base_elevation + (normalized * max_height)
        else:
            # Flat terrain
            heightmap = np.full_like(heightmap, base_elevation)
        
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
        
        # Initialize heightmap
        heightmap = np.zeros_like(X)
        
        # Apply base terrain
        if "base" in self.feature_generators:
            base_seed = seeds[0] if seeds else global_seed
            heightmap = self.feature_generators["base"].apply(
                heightmap, X, Y, parameters, base_seed
            )
        
        # Apply primary features
        primary_features = features.get("primary_features", [])
        for i, feature_name in enumerate(primary_features):
            if feature_name in self.feature_generators:
                seed = seeds[i + 1] if i + 1 < len(seeds) else global_seed + i + 1
                heightmap = self.feature_generators[feature_name].apply(
                    heightmap, X, Y, parameters, seed
                )
        
        # Apply secondary features  
        secondary_features = features.get("secondary_features", [])
        for i, feature_name in enumerate(secondary_features):
            if feature_name in self.feature_generators:
                seed = seeds[len(primary_features) + i + 1] if len(primary_features) + i + 1 < len(seeds) else global_seed + len(primary_features) + i + 1
                heightmap = self.feature_generators[feature_name].apply(
                    heightmap, X, Y, parameters, seed
                )
        
        # Apply realistic scaling (same as legacy)
        heightmap = np.clip(heightmap, 0, None)
        terrain_type = features.get("terrain_type", "plains")
        
        terrain_scales = {"plains": 50, "hills": 300, "valleys": 200, "mountains": 1200}
        base_elevations = {"plains": 5, "hills": 20, "valleys": 0, "mountains": 100}
        
        max_height = terrain_scales.get(terrain_type, 200)
        base_elevation = base_elevations.get(terrain_type, 10)
        
        if heightmap.max() > 1e-6:
            normalized = heightmap / heightmap.max()
            heightmap = base_elevation + (normalized * max_height)
        else:
            heightmap = np.full_like(heightmap, base_elevation)
        
        return heightmap
    
    def _generate_simple(
        self,
        world_x: int, world_y: int,
        parameters: Dict[str, float],
        global_seed: int
    ) -> np.ndarray:
        """Generate simple noise for fallback."""
        
        x_coords = np.linspace(world_x, world_x + self.tile_size, self.tile_size, endpoint=False)
        y_coords = np.linspace(world_y, world_y + self.tile_size, self.tile_size, endpoint=False)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        heightmap = self._simple_perlin(X, Y, global_seed, parameters)
        heightmap = np.clip(heightmap, 0, None)
        
        # Simple scaling
        if heightmap.max() > 1e-6:
            heightmap = (heightmap / heightmap.max()) * 100  # 0-100m
        
        return heightmap

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
        
        # Random gradients (simplified)
        np.random.seed(seed)
        rand_vals = np.random.random((x.shape[0] + 2, x.shape[1] + 2))
        
        # Sample noise at grid corners
        n00 = rand_vals[x0 % rand_vals.shape[0], y0 % rand_vals.shape[1]]
        n10 = rand_vals[x1 % rand_vals.shape[0], y0 % rand_vals.shape[1]]
        n01 = rand_vals[x0 % rand_vals.shape[0], y1 % rand_vals.shape[1]]
        n11 = rand_vals[x1 % rand_vals.shape[0], y1 % rand_vals.shape[1]]
        
        # Interpolate
        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        
        return nx0 * (1 - v) + nx1 * v
    
    def _ridged_noise(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """Ridged multifractal noise."""
        
        noise = self._simple_perlin(X, Y, seed, parameters)
        # Create ridges by taking absolute value and inverting
        ridged = 1.0 - np.abs(noise)
        return ridged
    
    def _domain_warp(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Simple domain warping effect."""
        
        warp_strength = parameters.get("warp_amplitude", 10.0)
        
        # Create coordinate grids
        h, w = heightmap.shape
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Generate warp offsets
        np.random.seed(seed)
        warp_x = np.random.random(heightmap.shape) * warp_strength - warp_strength/2
        warp_y = np.random.random(heightmap.shape) * warp_strength - warp_strength/2
        
        # Apply warping (simplified)
        warped = np.copy(heightmap)
        for i in range(h):
            for j in range(w):
                src_i = int(np.clip(i + warp_y[i, j], 0, h-1))
                src_j = int(np.clip(j + warp_x[i, j], 0, w-1))
                warped[i, j] = heightmap[src_i, src_j]
        
        return warped
    
    def _simple_erosion(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Simple erosion simulation."""
        
        erosion_strength = parameters.get("erosion_speed", 0.1)
        iterations = int(parameters.get("iterations", 10))
        
        eroded = np.copy(heightmap)
        
        for _ in range(iterations):
            # Simple smoothing-based erosion
            smoothed = self._smooth_heightmap(eroded)
            eroded = eroded * (1 - erosion_strength) + smoothed * erosion_strength
        
        return eroded
    
    def _smooth_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply smoothing filter to heightmap."""
        
        h, w = heightmap.shape
        smoothed = np.copy(heightmap)
        
        # Simple 3x3 averaging
        for i in range(1, h-1):
            for j in range(1, w-1):
                smoothed[i, j] = np.mean(heightmap[i-1:i+2, j-1:j+2])
        
        return smoothed