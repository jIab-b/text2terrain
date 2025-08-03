"""
Valley terrain generator.

Creates predictable valley features including deep channels, river valleys, and basins.
"""

import numpy as np
from typing import Dict
from .base import FeatureGenerator


class ValleyGenerator(FeatureGenerator):
    """
    Generates guaranteed valley features.
    
    Creates deep channels and basins that LLMs can reliably learn
    to associate with valley descriptions.
    """
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Carve valleys into existing terrain."""
        
        valley_depth = parameters.get("valley_depth", 0.6)
        valley_width = parameters.get("valley_width", 0.4)
        valley_count = int(parameters.get("valley_count", 2))
        meandering = parameters.get("valley_meandering", 0.3)
        
        # Generate valley network
        valleys = self._generate_valley_network(
            heightmap.shape, seed, valley_count, valley_width, meandering
        )
        
        # Apply depth with smooth falloff
        valley_carving = valleys * valley_depth
        
        # Carve valleys (subtract from heightmap)
        result = heightmap.copy()
        valley_mask = valley_carving > 0
        result[valley_mask] -= valley_carving[valley_mask]
        
        # Ensure non-negative values
        return np.maximum(result, 0)
    
    def _generate_valley_network(
        self,
        shape: tuple,
        seed: int,
        valley_count: int,
        width: float,
        meandering: float
    ) -> np.ndarray:
        """Generate network of valley channels."""
        
        rows, cols = shape
        valleys = np.zeros(shape)
        np.random.seed(seed)
        
        for i in range(valley_count):
            valley = self._generate_single_valley(
                shape, seed + i, width, meandering
            )
            valleys = np.maximum(valleys, valley)
        
        return valleys
    
    def _generate_single_valley(
        self,
        shape: tuple,
        seed: int,
        width: float,
        meandering: float
    ) -> np.ndarray:
        """Generate a single valley channel."""
        
        rows, cols = shape
        valley = np.zeros(shape)
        np.random.seed(seed)
        
        # Choose random start and end points
        if np.random.random() > 0.5:
            # Horizontal valley (left to right)
            start_y = np.random.uniform(0.2, 0.8) * rows
            end_y = start_y + np.random.uniform(-0.3, 0.3) * rows * meandering
            
            # Generate path
            for x in range(cols):
                progress = x / cols
                y = start_y + (end_y - start_y) * progress
                
                # Add meandering
                meander_offset = np.sin(progress * np.pi * 4) * meandering * 20
                y += meander_offset
                
                # Clamp to valid range
                y = np.clip(y, 0, rows - 1)
                
                # Create valley cross-section
                valley_radius = width * 30  # Convert to pixels
                self._add_valley_cross_section(valley, int(y), x, valley_radius, True)
        else:
            # Vertical valley (top to bottom)
            start_x = np.random.uniform(0.2, 0.8) * cols
            end_x = start_x + np.random.uniform(-0.3, 0.3) * cols * meandering
            
            # Generate path
            for y in range(rows):
                progress = y / rows
                x = start_x + (end_x - start_x) * progress
                
                # Add meandering
                meander_offset = np.sin(progress * np.pi * 4) * meandering * 20
                x += meander_offset
                
                # Clamp to valid range
                x = np.clip(x, 0, cols - 1)
                
                # Create valley cross-section
                valley_radius = width * 30  # Convert to pixels
                self._add_valley_cross_section(valley, y, int(x), valley_radius, False)
        
        return valley
    
    def _add_valley_cross_section(
        self,
        valley: np.ndarray,
        center_y: int,
        center_x: int,
        radius: float,
        horizontal: bool
    ) -> None:
        """Add U-shaped valley cross-section at given location."""
        
        rows, cols = valley.shape
        
        if horizontal:
            # Valley runs horizontally, cross-section is vertical
            for dy in range(-int(radius), int(radius) + 1):
                y = center_y + dy
                if 0 <= y < rows:
                    distance = abs(dy) / radius
                    if distance <= 1.0:
                        # U-shaped profile
                        depth = 1.0 - distance**2
                        valley[y, center_x] = max(valley[y, center_x], depth)
        else:
            # Valley runs vertically, cross-section is horizontal
            for dx in range(-int(radius), int(radius) + 1):
                x = center_x + dx
                if 0 <= x < cols:
                    distance = abs(dx) / radius
                    if distance <= 1.0:
                        # U-shaped profile
                        depth = 1.0 - distance**2
                        valley[center_y, x] = max(valley[center_y, x], depth)
    
    def enhance(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add subtle valley enhancements as secondary feature."""
        
        enhancement_strength = parameters.get("valley_enhancement", 0.1)
        
        if enhancement_strength == 0:
            return heightmap
        
        # Add subtle erosion patterns in low areas
        rows, cols = heightmap.shape
        
        # Identify low areas (potential valleys)
        low_threshold = np.percentile(heightmap, 40)
        low_areas = heightmap < low_threshold
        
        # Add subtle noise to create more realistic valley texture
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Generate erosion noise
        erosion_noise = self._noise2d(X * 0.08, Y * 0.08, seed)
        enhancement = erosion_noise * enhancement_strength
        
        # Apply enhancement only to low areas
        result = heightmap.copy()
        result[low_areas] += enhancement[low_areas]
        
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