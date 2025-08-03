"""
River system generator.

Creates predictable river networks, lakes, and water features.
"""

import numpy as np
from typing import Dict
from .base import FeatureGenerator


class RiverGenerator(FeatureGenerator):
    """
    Generates river networks and water features.
    
    Creates flowing water systems that follow natural drainage patterns.
    """
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add river systems to terrain."""
        
        river_depth = parameters.get("river_depth", 0.3)
        river_width = parameters.get("river_width", 0.2)
        river_count = int(parameters.get("river_count", 2))
        lake_count = int(parameters.get("lake_count", 1))
        
        result = heightmap.copy()
        
        # Generate rivers (follow height gradients)
        for i in range(river_count):
            river = self._generate_river(
                heightmap, seed + i, river_width, river_depth
            )
            result = self._apply_water_feature(result, river)
        
        # Generate lakes (in low areas)
        for i in range(lake_count):
            lake = self._generate_lake(
                heightmap, seed + 100 + i, river_depth * 1.5
            )
            result = self._apply_water_feature(result, lake)
        
        return result
    
    def _generate_river(
        self,
        heightmap: np.ndarray,
        seed: int,
        width: float,
        depth: float
    ) -> np.ndarray:
        """Generate river following natural drainage."""
        
        rows, cols = heightmap.shape
        river = np.zeros_like(heightmap)
        np.random.seed(seed)
        
        # Find high point to start river
        high_areas = heightmap > np.percentile(heightmap, 80)
        high_indices = np.where(high_areas)
        
        if len(high_indices[0]) == 0:
            return river
        
        # Random starting point in high area
        start_idx = np.random.randint(len(high_indices[0]))
        current_y = high_indices[0][start_idx]
        current_x = high_indices[1][start_idx]
        
        # Follow steepest descent
        visited = set()
        river_width_pixels = max(3, int(width * 30))
        
        for step in range(min(rows, cols) * 2):  # Prevent infinite loops
            if (current_y, current_x) in visited:
                break
            
            visited.add((current_y, current_x))
            
            # Add river segment
            self._add_river_segment(river, current_y, current_x, river_width_pixels, depth)
            
            # Find steepest descent direction
            next_y, next_x = self._find_steepest_descent(
                heightmap, current_y, current_x
            )
            
            if next_y == current_y and next_x == current_x:
                # Reached minimum, stop
                break
            
            current_y, current_x = next_y, next_x
        
        return river
    
    def _find_steepest_descent(
        self,
        heightmap: np.ndarray,
        y: int, x: int
    ) -> tuple:
        """Find direction of steepest descent from current position."""
        
        rows, cols = heightmap.shape
        current_height = heightmap[y, x]
        
        # Check 8-connected neighbors
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        best_y, best_x = y, x
        steepest_drop = 0
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < rows and 0 <= nx < cols:
                neighbor_height = heightmap[ny, nx]
                drop = current_height - neighbor_height
                
                if drop > steepest_drop:
                    steepest_drop = drop
                    best_y, best_x = ny, nx
        
        return best_y, best_x
    
    def _add_river_segment(
        self,
        river: np.ndarray,
        center_y: int,
        center_x: int,
        width: int,
        depth: float
    ) -> None:
        """Add river segment with given width."""
        
        rows, cols = river.shape
        
        # Create circular river cross-section
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                y, x = center_y + dy, center_x + dx
                
                if 0 <= y < rows and 0 <= x < cols:
                    distance = np.sqrt(dy*dy + dx*dx)
                    if distance <= width:
                        # Smooth falloff
                        river_strength = depth * (1.0 - distance / width)
                        river[y, x] = max(river[y, x], river_strength)
    
    def _generate_lake(
        self,
        heightmap: np.ndarray,
        seed: int,
        depth: float
    ) -> np.ndarray:
        """Generate lake in low-lying area."""
        
        rows, cols = heightmap.shape
        lake = np.zeros_like(heightmap)
        np.random.seed(seed)
        
        # Find low areas for lake placement
        low_threshold = np.percentile(heightmap, 30)
        low_areas = heightmap < low_threshold
        low_indices = np.where(low_areas)
        
        if len(low_indices[0]) == 0:
            return lake
        
        # Choose random low point
        lake_idx = np.random.randint(len(low_indices[0]))
        center_y = low_indices[0][lake_idx]
        center_x = low_indices[1][lake_idx]
        
        # Create irregular lake shape
        lake_radius = np.random.uniform(15, 40)
        
        y_coords, x_coords = np.ogrid[:rows, :cols]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Add irregularity with noise
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')
        noise = self._noise2d(x_grid * 0.03, y_grid * 0.03, seed)
        irregular_radius = lake_radius * (1.0 + noise * 0.4)
        
        # Create lake mask
        lake_mask = distance < irregular_radius
        
        # Smooth depth falloff
        falloff_distance = distance / irregular_radius
        falloff_values = np.maximum(0, 1.0 - falloff_distance**1.5)
        
        lake[lake_mask] = falloff_values[lake_mask] * depth
        
        return lake
    
    def _apply_water_feature(
        self,
        heightmap: np.ndarray,
        water_feature: np.ndarray
    ) -> np.ndarray:
        """Apply water feature to heightmap (carve out water areas)."""
        
        result = heightmap.copy()
        water_mask = water_feature > 0
        result[water_mask] -= water_feature[water_mask]
        
        return np.maximum(result, 0)
    
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
        """Add subtle water erosion effects as secondary feature."""
        
        enhancement_strength = parameters.get("water_enhancement", 0.1)
        
        if enhancement_strength == 0:
            return heightmap
        
        # Simulate water erosion effects
        rows, cols = heightmap.shape
        
        # Calculate gradients to find water flow paths
        grad_y, grad_x = np.gradient(heightmap)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Areas with high gradient are erosion-prone
        erosion_areas = gradient_magnitude > np.percentile(gradient_magnitude, 70)
        
        # Add subtle erosion noise
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        erosion_noise = self._noise2d(X * 0.06, Y * 0.06, seed)
        enhancement = erosion_noise * enhancement_strength
        
        # Apply erosion to steep areas
        result = heightmap.copy()
        result[erosion_areas] += enhancement[erosion_areas]
        
        return result