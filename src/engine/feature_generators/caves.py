"""
Cave system generator.

Creates underground cave networks and caverns for Minecraft-level terrain complexity.
"""

import numpy as np
from typing import Dict
from .base import FeatureGenerator


class CaveGenerator(FeatureGenerator):
    """
    Generates cave systems and underground features.
    
    Creates predictable cave networks that can be learned by LLMs
    for underground terrain generation.
    """
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add cave systems to terrain (represented as negative height)."""
        
        cave_density = parameters.get("cave_density", 0.3)
        cave_size = parameters.get("cave_size", 0.5)
        tunnel_width = parameters.get("tunnel_width", 0.3)
        depth_range = parameters.get("cave_depth", 0.4)
        
        # Generate cave network
        caves = self._generate_cave_network(
            heightmap.shape, seed, cave_density, cave_size, tunnel_width
        )
        
        # Apply depth variation
        cave_depths = caves * depth_range
        
        # Create cave effect (subtract from terrain in elevated areas only)
        result = heightmap.copy()
        
        # Only create caves in areas with sufficient height
        min_height_for_caves = np.percentile(heightmap, 30)
        cave_areas = (caves > 0) & (heightmap > min_height_for_caves)
        
        result[cave_areas] -= cave_depths[cave_areas]
        
        return np.maximum(result, 0)
    
    def _generate_cave_network(
        self,
        shape: tuple,
        seed: int,
        density: float,
        size: float,
        tunnel_width: float
    ) -> np.ndarray:
        """Generate network of connected caves and tunnels."""
        
        rows, cols = shape
        caves = np.zeros(shape)
        np.random.seed(seed)
        
        # Generate cave nodes (large caverns)
        num_caves = max(1, int(density * 8))
        cave_centers = []
        
        for i in range(num_caves):
            center_x = np.random.uniform(0.2, 0.8) * cols
            center_y = np.random.uniform(0.2, 0.8) * rows
            cave_centers.append((center_x, center_y))
            
            # Create cavern
            cavern = self._create_cavern(
                shape, center_x, center_y, size * 40, seed + i
            )
            caves = np.maximum(caves, cavern)
        
        # Connect caves with tunnels
        for i in range(len(cave_centers) - 1):
            start = cave_centers[i]
            end = cave_centers[i + 1]
            
            tunnel = self._create_tunnel(
                shape, start, end, tunnel_width * 15, seed + 100 + i
            )
            caves = np.maximum(caves, tunnel)
        
        # Add some random tunnels for complexity
        num_extra_tunnels = max(1, int(density * 3))
        for i in range(num_extra_tunnels):
            start_x = np.random.uniform(0.1, 0.9) * cols
            start_y = np.random.uniform(0.1, 0.9) * rows
            end_x = np.random.uniform(0.1, 0.9) * cols
            end_y = np.random.uniform(0.1, 0.9) * rows
            
            tunnel = self._create_tunnel(
                shape, (start_x, start_y), (end_x, end_y),
                tunnel_width * 10, seed + 200 + i
            )
            caves = np.maximum(caves, tunnel)
        
        return caves
    
    def _create_cavern(
        self,
        shape: tuple,
        center_x: float,
        center_y: float,
        radius: float,
        seed: int
    ) -> np.ndarray:
        """Create a large cavern (irregular circular area)."""
        
        rows, cols = shape
        cavern = np.zeros(shape)
        
        # Create distance field
        y_coords, x_coords = np.ogrid[:rows, :cols]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Add irregularity using noise
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')
        noise = self._noise2d(x_grid * 0.05, y_grid * 0.05, seed)
        irregular_radius = radius * (1.0 + noise * 0.3)
        
        # Create cavern mask
        cavern_mask = distance < irregular_radius
        
        # Smooth falloff at edges
        falloff_distance = distance / irregular_radius
        falloff_values = np.maximum(0, 1.0 - falloff_distance**2)
        
        cavern[cavern_mask] = falloff_values[cavern_mask]
        
        return cavern
    
    def _create_tunnel(
        self,
        shape: tuple,
        start: tuple,
        end: tuple,
        width: float,
        seed: int
    ) -> np.ndarray:
        """Create tunnel between two points."""
        
        rows, cols = shape
        tunnel = np.zeros(shape)
        
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate tunnel path
        num_segments = max(10, int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)))
        
        for i in range(num_segments):
            progress = i / (num_segments - 1)
            
            # Linear interpolation with some wandering
            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress
            
            # Add some random wandering
            np.random.seed(seed + i)
            wander_x = np.random.uniform(-width * 0.5, width * 0.5)
            wander_y = np.random.uniform(-width * 0.5, width * 0.5)
            
            current_x += wander_x
            current_y += wander_y
            
            # Clamp to valid coordinates
            current_x = np.clip(current_x, 0, cols - 1)
            current_y = np.clip(current_y, 0, rows - 1)
            
            # Add circular tunnel segment
            self._add_tunnel_segment(tunnel, current_x, current_y, width)
        
        return tunnel
    
    def _add_tunnel_segment(
        self,
        tunnel: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float
    ) -> None:
        """Add circular tunnel segment at given location."""
        
        rows, cols = tunnel.shape
        
        # Define circular area
        y_coords, x_coords = np.ogrid[:rows, :cols]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Smooth circular falloff
        mask = distance < radius
        falloff_values = np.maximum(0, 1.0 - (distance / radius)**2)
        
        # Add to tunnel (take maximum to avoid overwriting)
        tunnel[mask] = np.maximum(tunnel[mask], falloff_values[mask])
    
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
        """Add subtle cave enhancements as secondary feature."""
        
        enhancement_strength = parameters.get("cave_enhancement", 0.05)
        
        if enhancement_strength == 0:
            return heightmap
        
        # Add subtle underground texture effects
        rows, cols = heightmap.shape
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # High-frequency underground noise
        underground_noise = self._noise2d(X * 0.15, Y * 0.15, seed)
        enhancement = underground_noise * enhancement_strength
        
        # Apply only to mid-height areas (where caves might be)
        mid_height_mask = (heightmap > np.percentile(heightmap, 25)) & \
                         (heightmap < np.percentile(heightmap, 75))
        
        result = heightmap.copy()
        result[mid_height_mask] += enhancement[mid_height_mask]
        
        return result