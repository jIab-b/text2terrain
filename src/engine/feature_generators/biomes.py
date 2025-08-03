"""
Biome system generator.

Creates biome-specific terrain modifications and environmental features.
"""

import numpy as np
from typing import Dict
from .base import FeatureGenerator


class BiomeGenerator(FeatureGenerator):
    """
    Generates biome-specific terrain modifications.
    
    Applies environmental effects based on biome type (desert, forest, arctic, etc.).
    """
    
    BIOME_TYPES = {
        "plains": {"roughness": 0.1, "elevation_mod": 0.0, "water_mod": 0.0},
        "desert": {"roughness": 0.3, "elevation_mod": -0.1, "water_mod": -0.5},
        "forest": {"roughness": 0.2, "elevation_mod": 0.1, "water_mod": 0.2},
        "mountain": {"roughness": 0.8, "elevation_mod": 0.5, "water_mod": -0.2},
        "arctic": {"roughness": 0.1, "elevation_mod": 0.0, "water_mod": -0.8},
        "swamp": {"roughness": 0.4, "elevation_mod": -0.3, "water_mod": 0.8},
        "volcanic": {"roughness": 0.9, "elevation_mod": 0.3, "water_mod": -0.3}
    }
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Apply biome-specific terrain modifications."""
        
        biome_type = parameters.get("biome_type", "plains")
        biome_strength = parameters.get("biome_strength", 0.5)
        
        if biome_type not in self.BIOME_TYPES:
            biome_type = "plains"
        
        biome_config = self.BIOME_TYPES[biome_type]
        
        result = heightmap.copy()
        
        # Apply biome-specific modifications
        if biome_type == "desert":
            result = self._apply_desert_features(result, biome_config, biome_strength, seed)
        elif biome_type == "forest":
            result = self._apply_forest_features(result, biome_config, biome_strength, seed)
        elif biome_type == "mountain":
            result = self._apply_mountain_features(result, biome_config, biome_strength, seed)
        elif biome_type == "arctic":
            result = self._apply_arctic_features(result, biome_config, biome_strength, seed)
        elif biome_type == "swamp":
            result = self._apply_swamp_features(result, biome_config, biome_strength, seed)
        elif biome_type == "volcanic":
            result = self._apply_volcanic_features(result, biome_config, biome_strength, seed)
        else:  # plains
            result = self._apply_plains_features(result, biome_config, biome_strength, seed)
        
        return result
    
    def _apply_desert_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply desert-specific terrain features."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Create sand dunes using sine waves
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Large dune patterns
        dune_freq = 0.02
        dune_pattern = np.sin(X * dune_freq) * np.sin(Y * dune_freq * 0.7)
        dune_height = config["roughness"] * strength * 0.3
        
        # Add to low-lying areas (desert dunes form in flats)
        flat_threshold = np.percentile(heightmap, 60)
        flat_areas = heightmap < flat_threshold
        result[flat_areas] += dune_pattern[flat_areas] * dune_height
        
        # Add rocky outcrops in high areas
        rocky_noise = self._noise2d(X * 0.05, Y * 0.05, seed)
        rocky_areas = heightmap > np.percentile(heightmap, 80)
        result[rocky_areas] += rocky_noise[rocky_areas] * config["roughness"] * strength * 0.2
        
        return result
    
    def _apply_forest_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply forest-specific terrain features."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Create subtle hill variations (root systems and tree growth)
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Fine detail noise for forest undergrowth
        forest_noise = self._noise2d(X * 0.08, Y * 0.08, seed)
        forest_detail = forest_noise * config["roughness"] * strength * 0.15
        
        result += forest_detail
        
        # Create clearings (small flat areas)
        clearing_centers = 3
        np.random.seed(seed)
        
        for i in range(clearing_centers):
            center_x = np.random.uniform(0.2, 0.8) * cols
            center_y = np.random.uniform(0.2, 0.8) * rows
            radius = np.random.uniform(10, 25)
            
            # Smooth circular clearings
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            clearing_mask = distance < radius
            smoothing_strength = 1.0 - distance / radius
            smoothing_strength = np.maximum(0, smoothing_strength)
            
            # Smooth terrain in clearings
            avg_height = np.mean(result[clearing_mask]) if np.any(clearing_mask) else 0
            result[clearing_mask] = (result[clearing_mask] * 0.3 + 
                                   avg_height * 0.7 * smoothing_strength[clearing_mask])
        
        return result
    
    def _apply_mountain_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply mountain biome features (already handled by MountainGenerator usually)."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Add high-altitude effects - more rugged terrain at elevation
        high_areas = heightmap > np.percentile(heightmap, 70)
        
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # High-altitude roughness
        alpine_noise = self._noise2d(X * 0.12, Y * 0.12, seed)
        alpine_effect = alpine_noise * config["roughness"] * strength * 0.2
        
        result[high_areas] += alpine_effect[high_areas]
        
        return result
    
    def _apply_arctic_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply arctic/tundra terrain features."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Smooth terrain (ice and snow cover)
        # Apply Gaussian blur effect to simulate snow accumulation
        from scipy import ndimage
        
        # Smooth the terrain slightly
        smoothed = ndimage.gaussian_filter(result, sigma=strength * 2.0)
        result = result * (1 - strength * 0.3) + smoothed * (strength * 0.3)
        
        # Add polygonal tundra patterns
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Hexagonal-ish patterns (permafrost polygons)
        poly_freq = 0.03
        poly_pattern = (np.sin(X * poly_freq) + 
                       np.sin(X * poly_freq * 0.5 + Y * poly_freq * 0.866) +
                       np.sin(-X * poly_freq * 0.5 + Y * poly_freq * 0.866))
        
        poly_effect = poly_pattern * config["roughness"] * strength * 0.1
        result += poly_effect
        
        return result
    
    def _apply_swamp_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply swamp/wetland terrain features."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Lower overall elevation
        result += config["elevation_mod"] * strength
        
        # Create small water pools and raised hummocks
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Swamp texture noise
        swamp_noise = self._noise2d(X * 0.06, Y * 0.06, seed)
        
        # Create alternating high/low areas
        swamp_pattern = np.where(swamp_noise > 0, 
                                swamp_noise * 0.2,  # Raised hummocks
                                swamp_noise * 0.4)  # Water pools
        
        swamp_effect = swamp_pattern * config["roughness"] * strength
        result += swamp_effect
        
        return result
    
    def _apply_volcanic_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply volcanic terrain features."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Create volcanic cones and lava flows
        np.random.seed(seed)
        
        # Add a few volcanic cones
        num_cones = max(1, int(strength * 3))
        
        for i in range(num_cones):
            cone_x = np.random.uniform(0.2, 0.8) * cols
            cone_y = np.random.uniform(0.2, 0.8) * rows
            cone_radius = np.random.uniform(20, 50)
            cone_height = config["elevation_mod"] * strength
            
            # Create conical elevation
            x_coords = np.arange(cols)
            y_coords = np.arange(rows)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            distance = np.sqrt((X - cone_x)**2 + (Y - cone_y)**2)
            cone_mask = distance < cone_radius
            cone_elevation = np.maximum(0, cone_height * (1 - distance / cone_radius))
            
            result[cone_mask] += cone_elevation[cone_mask]
        
        # Add rugged volcanic texture
        volcanic_noise = self._noise2d(X * 0.1, Y * 0.1, seed + 100)
        volcanic_texture = volcanic_noise * config["roughness"] * strength * 0.3
        result += volcanic_texture
        
        return result
    
    def _apply_plains_features(
        self,
        heightmap: np.ndarray,
        config: Dict,
        strength: float,
        seed: int
    ) -> np.ndarray:
        """Apply plains terrain features (gentle rolling hills)."""
        
        rows, cols = heightmap.shape
        result = heightmap.copy()
        
        # Very gentle rolling hills
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Large-scale gentle undulations
        plains_noise = self._noise2d(X * 0.01, Y * 0.01, seed)
        plains_effect = plains_noise * config["roughness"] * strength * 0.5
        
        result += plains_effect
        
        return result
    
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
        
        # Hash function
        def hash2d(ix, iy):
            h = (ix * 374761393 + iy * 668265263 + seed * 1664525) % 2147483647
            return (h / 2147483647.0) * 2.0 - 1.0
        
        # Corner values
        c00 = hash2d(x0, y0)
        c10 = hash2d(x1, y0)
        c01 = hash2d(x0, y1)
        c11 = hash2d(x1, y1)
        
        # Bilinear interpolation
        top = c00 + u * (c10 - c00)
        bottom = c01 + u * (c11 - c01)
        
        return top + v * (bottom - top)
    
    def enhance(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add subtle biome enhancements as secondary feature."""
        
        biome_type = parameters.get("biome_type", "plains")
        enhancement_strength = parameters.get("biome_enhancement", 0.1)
        
        if enhancement_strength == 0:
            return heightmap
        
        # Add subtle environmental effects based on biome
        rows, cols = heightmap.shape
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Biome-specific detail noise
        if biome_type in ["desert", "volcanic"]:
            # High-frequency detail for rough biomes
            detail_noise = self._noise2d(X * 0.15, Y * 0.15, seed)
        else:
            # Lower-frequency detail for smoother biomes  
            detail_noise = self._noise2d(X * 0.08, Y * 0.08, seed)
        
        enhancement = detail_noise * enhancement_strength
        
        return heightmap + enhancement