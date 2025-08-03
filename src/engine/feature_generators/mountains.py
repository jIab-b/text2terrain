"""
Mountain terrain generator.

Creates predictable mountain features including peaks, ridges, and slopes.
"""

import numpy as np
from typing import Dict
from .base import FeatureGenerator


class MountainGenerator(FeatureGenerator):
    """
    Generates guaranteed mountain features.
    
    Creates steep terrain with identifiable peaks and ridges
    that LLMs can reliably learn to associate with mountain descriptions.
    """
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add mountain features to existing terrain."""
        
        mountain_height = parameters.get("mountain_height", 0.8)
        mountain_steepness = parameters.get("mountain_steepness", 0.6)
        peak_count = int(parameters.get("peak_count", 3))
        ridge_prominence = parameters.get("ridge_prominence", 0.7)
        
        # Generate mountain base shape
        mountain_base = self._generate_mountain_base(X, Y, seed, mountain_height)
        
        # Add peaks
        peaks = self._generate_peaks(X, Y, seed + 1, peak_count, mountain_height)
        
        # Add ridges
        ridges = self._generate_ridges(X, Y, seed + 2, ridge_prominence)
        
        # Apply steepness transformation
        mountain_terrain = self._apply_steepness(
            mountain_base + peaks + ridges, mountain_steepness
        )
        
        # Blend with existing terrain
        mountain_mask = mountain_terrain > 0.2  # Only elevate significant areas
        result = heightmap.copy()
        result[mountain_mask] += mountain_terrain[mountain_mask]
        
        return result
    
    def _generate_mountain_base(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        height: float
    ) -> np.ndarray:
        """Generate base mountain shape using multiple noise octaves."""
        
        np.random.seed(seed)
        
        # Large-scale mountain shape
        freq_base = 0.005
        base_shape = self._noise2d(X * freq_base, Y * freq_base, seed)
        
        # Medium detail
        freq_med = 0.02
        medium_detail = self._noise2d(X * freq_med, Y * freq_med, seed + 100)
        
        # Combine for mountain base
        mountain_base = (base_shape * 0.7 + medium_detail * 0.3) * height
        
        # Ensure positive values only (mountains go up)
        return np.maximum(mountain_base, 0)
    
    def _generate_peaks(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        peak_count: int,
        height: float
    ) -> np.ndarray:
        """Generate distinct mountain peaks."""
        
        np.random.seed(seed)
        rows, cols = X.shape
        peaks = np.zeros_like(X)
        
        # Generate peak locations
        for i in range(peak_count):
            # Random peak center
            peak_x = np.random.uniform(0.2, 0.8) * cols
            peak_y = np.random.uniform(0.2, 0.8) * rows
            
            # Peak radius and height
            radius = np.random.uniform(20, 60)
            peak_height = height * np.random.uniform(0.8, 1.2)
            
            # Create peak using distance function
            dx = (np.arange(cols) - peak_x)
            dy = (np.arange(rows) - peak_y)
            DX, DY = np.meshgrid(dx, dy, indexing='ij')
            
            distance = np.sqrt(DX**2 + DY**2)
            peak = np.maximum(0, peak_height * (1 - distance / radius))
            
            # Add to peaks array
            peaks = np.maximum(peaks, peak)
        
        return peaks
    
    def _generate_ridges(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        prominence: float
    ) -> np.ndarray:
        """Generate mountain ridges using ridged noise."""
        
        # High-frequency ridged noise
        freq = 0.03
        noise = self._noise2d(X * freq, Y * freq, seed)
        
        # Create ridges by inverting absolute values
        ridged = 1.0 - np.abs(noise)
        ridged = np.power(ridged, 2.0)  # Sharpen ridges
        
        return ridged * prominence
    
    def _apply_steepness(
        self,
        terrain: np.ndarray,
        steepness: float
    ) -> np.ndarray:
        """Apply steepness transformation to create sharp mountain features."""
        
        # Steepness controls how sharp the mountains are
        # Higher steepness = sharper, more dramatic peaks
        power = 1.0 + steepness * 2.0
        return np.power(terrain, 1.0 / power)
    
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
        """Add subtle mountain enhancements as secondary feature."""
        
        enhancement_strength = parameters.get("mountain_enhancement", 0.2)
        
        if enhancement_strength == 0:
            return heightmap
        
        # Add subtle noise to create more realistic mountain texture
        rows, cols = heightmap.shape
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # High-frequency detail noise
        detail_noise = self._noise2d(X * 0.1, Y * 0.1, seed)
        enhancement = detail_noise * enhancement_strength
        
        # Only enhance elevated areas (mountains)
        mountain_mask = heightmap > np.percentile(heightmap, 60)
        result = heightmap.copy()
        result[mountain_mask] += enhancement[mountain_mask]
        
        return result