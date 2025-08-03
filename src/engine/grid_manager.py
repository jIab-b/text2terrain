"""
Grid manager for seamless terrain generation.

Handles tile-based terrain generation with continuous boundaries
for infinite world exploration.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import ndimage


class GridManager:
    """
    Manages seamless grid generation and boundary smoothing.
    
    Ensures terrain tiles connect smoothly for infinite world generation
    while maintaining feature consistency across boundaries.
    """
    
    def __init__(self, tile_size: int = 256, overlap: int = 16):
        self.tile_size = tile_size
        self.overlap = overlap
        self.boundary_cache = {}
    
    def generate_seamless_region(
        self,
        center_x: int, center_y: int,
        grid_size: int,
        terrain_composer,
        **kwargs
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Generate seamless grid of terrain tiles.
        
        Args:
            center_x: Center tile X coordinate  
            center_y: Center tile Y coordinate
            grid_size: Size of grid (e.g., 3 = 3x3 grid)
            terrain_composer: TerrainComposer instance for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping (tile_x, tile_y) to heightmap arrays
        """
        
        tiles = {}
        half_grid = grid_size // 2
        
        # Phase 1: Generate all tiles with extended boundaries
        for dy in range(-half_grid, half_grid + 1):
            for dx in range(-half_grid, half_grid + 1):
                tile_x = center_x + dx
                tile_y = center_y + dy
                
                world_x = tile_x * self.tile_size
                world_y = tile_y * self.tile_size
                
                # Generate tile with overlap for seamless blending
                tiles[(tile_x, tile_y)] = self._generate_tile_with_overlap(
                    terrain_composer, world_x, world_y, **kwargs
                )
        
        # Phase 2: Apply boundary smoothing
        smoothed_tiles = {}
        for (tile_x, tile_y), heightmap in tiles.items():
            neighbors = self._get_tile_neighbors(tiles, tile_x, tile_y)
            smoothed_tiles[(tile_x, tile_y)] = self._apply_boundary_smoothing(
                heightmap, neighbors
            )
        
        return smoothed_tiles
    
    def _generate_tile_with_overlap(
        self,
        terrain_composer,
        world_x: int, world_y: int,
        **kwargs
    ) -> np.ndarray:
        """Generate tile with extended boundaries for seamless blending."""
        
        # Generate slightly larger tile
        extended_size = self.tile_size + 2 * self.overlap
        extended_world_x = world_x - self.overlap
        extended_world_y = world_y - self.overlap
        
        # Temporarily override tile size for extended generation
        original_tile_size = terrain_composer.tile_size
        terrain_composer.tile_size = extended_size
        
        # Generate extended heightmap
        extended_heightmap = terrain_composer.generate_heightmap(
            extended_world_x, extended_world_y, **kwargs
        )
        
        # Restore original tile size
        terrain_composer.tile_size = original_tile_size
        
        # Extract center portion (original tile size)
        start_idx = self.overlap
        end_idx = start_idx + self.tile_size
        
        if extended_heightmap.shape[0] >= end_idx and extended_heightmap.shape[1] >= end_idx:
            return extended_heightmap[start_idx:end_idx, start_idx:end_idx]
        else:
            # Fallback to regular generation if extended generation failed
            return terrain_composer.generate_heightmap(world_x, world_y, **kwargs)
    
    def _get_tile_neighbors(
        self,
        tiles: Dict[Tuple[int, int], np.ndarray],
        tile_x: int, tile_y: int
    ) -> Dict[str, Optional[np.ndarray]]:
        """Get neighboring tiles for boundary smoothing."""
        
        neighbors = {
            'north': tiles.get((tile_x, tile_y - 1)),
            'south': tiles.get((tile_x, tile_y + 1)), 
            'east': tiles.get((tile_x + 1, tile_y)),
            'west': tiles.get((tile_x - 1, tile_y)),
            'northeast': tiles.get((tile_x + 1, tile_y - 1)),
            'northwest': tiles.get((tile_x - 1, tile_y - 1)),
            'southeast': tiles.get((tile_x + 1, tile_y + 1)),
            'southwest': tiles.get((tile_x - 1, tile_y + 1))
        }
        
        return neighbors
    
    def apply_boundary_smoothing(
        self,
        heightmap: np.ndarray,
        world_x: int, world_y: int,
        grid_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply boundary smoothing for seamless transitions.
        
        Args:
            heightmap: Input heightmap
            world_x: World X coordinate
            world_y: World Y coordinate  
            grid_info: Grid continuity configuration
            
        Returns:
            Smoothed heightmap
        """
        
        if not grid_info.get("blend_edges", False):
            return heightmap
        
        result = heightmap.copy()
        
        # Apply boundary constraints if provided
        if "boundary_constraints" in grid_info:
            result = self._apply_boundary_constraints(
                result, grid_info["boundary_constraints"]
            )
        
        # Apply edge smoothing
        blend_width = grid_info.get("overlap_size", self.overlap)
        result = self._smooth_edges(result, blend_width)
        
        return result
    
    def _apply_boundary_smoothing(
        self,
        heightmap: np.ndarray,
        neighbors: Dict[str, Optional[np.ndarray]]
    ) -> np.ndarray:
        """Apply smoothing between adjacent tiles."""
        
        result = heightmap.copy()
        blend_width = min(8, self.overlap)
        
        # Smooth north edge
        if neighbors['north'] is not None:
            north_tile = neighbors['north']
            if north_tile.shape == heightmap.shape:
                # Get south edge of north neighbor
                north_edge = north_tile[-blend_width:, :]
                # Blend with north edge of current tile
                for i in range(blend_width):
                    alpha = (i + 1) / (blend_width + 1)
                    result[i, :] = (1 - alpha) * north_edge[i, :] + alpha * result[i, :]
        
        # Smooth south edge
        if neighbors['south'] is not None:
            south_tile = neighbors['south']
            if south_tile.shape == heightmap.shape:
                # Get north edge of south neighbor
                south_edge = south_tile[:blend_width, :]
                # Blend with south edge of current tile
                for i in range(blend_width):
                    alpha = (i + 1) / (blend_width + 1)
                    idx = -(blend_width - i)
                    result[idx, :] = (1 - alpha) * south_edge[i, :] + alpha * result[idx, :]
        
        # Smooth east edge
        if neighbors['east'] is not None:
            east_tile = neighbors['east']
            if east_tile.shape == heightmap.shape:
                # Get west edge of east neighbor
                east_edge = east_tile[:, :blend_width]
                # Blend with east edge of current tile
                for i in range(blend_width):
                    alpha = (i + 1) / (blend_width + 1)
                    idx = -(blend_width - i)
                    result[:, idx] = (1 - alpha) * east_edge[:, i] + alpha * result[:, idx]
        
        # Smooth west edge  
        if neighbors['west'] is not None:
            west_tile = neighbors['west']
            if west_tile.shape == heightmap.shape:
                # Get east edge of west neighbor
                west_edge = west_tile[:, -blend_width:]
                # Blend with west edge of current tile
                for i in range(blend_width):
                    alpha = (i + 1) / (blend_width + 1)
                    result[:, i] = (1 - alpha) * west_edge[:, i] + alpha * result[:, i]
        
        return result
    
    def _apply_boundary_constraints(
        self,
        heightmap: np.ndarray,
        constraints: Dict[str, List[float]]
    ) -> np.ndarray:
        """Apply specific height constraints at tile boundaries."""
        
        result = heightmap.copy()
        rows, cols = heightmap.shape
        
        # Apply constraints with smooth falloff
        falloff_width = min(8, self.overlap)
        
        # North edge constraint
        if "north" in constraints:
            target_values = np.array(constraints["north"])
            if len(target_values) == cols:
                for i in range(falloff_width):
                    weight = np.exp(-i * 0.5)  # Exponential falloff
                    result[i, :] = (1 - weight) * result[i, :] + weight * target_values
        
        # South edge constraint
        if "south" in constraints:
            target_values = np.array(constraints["south"])
            if len(target_values) == cols:
                for i in range(falloff_width):
                    weight = np.exp(-i * 0.5)
                    idx = rows - 1 - i
                    result[idx, :] = (1 - weight) * result[idx, :] + weight * target_values
        
        # East edge constraint
        if "east" in constraints:
            target_values = np.array(constraints["east"])
            if len(target_values) == rows:
                for i in range(falloff_width):
                    weight = np.exp(-i * 0.5)
                    idx = cols - 1 - i
                    result[:, idx] = (1 - weight) * result[:, idx] + weight * target_values
        
        # West edge constraint
        if "west" in constraints:
            target_values = np.array(constraints["west"])
            if len(target_values) == rows:
                for i in range(falloff_width):
                    weight = np.exp(-i * 0.5)
                    result[:, i] = (1 - weight) * result[:, i] + weight * target_values
        
        return result
    
    def _smooth_edges(self, heightmap: np.ndarray, blend_width: int) -> np.ndarray:
        """Apply general edge smoothing to reduce discontinuities."""
        
        if blend_width <= 0:
            return heightmap
        
        result = heightmap.copy()
        
        # Apply Gaussian smoothing near edges
        smoothed = ndimage.gaussian_filter(result, sigma=1.0)
        
        # Create edge mask with smooth falloff
        rows, cols = heightmap.shape
        edge_mask = np.zeros((rows, cols))
        
        # Distance from edges
        y_coords, x_coords = np.ogrid[:rows, :cols]
        
        # Distance to nearest edge
        dist_to_edge = np.minimum(
            np.minimum(x_coords, cols - 1 - x_coords),
            np.minimum(y_coords, rows - 1 - y_coords)
        )
        
        # Smooth blending near edges
        edge_blend = np.exp(-dist_to_edge / blend_width)
        edge_blend = np.clip(edge_blend, 0, 1)
        
        # Apply smoothing with falloff
        result = (1 - edge_blend) * result + edge_blend * smoothed
        
        return result
    
    def get_boundary_values(
        self,
        heightmap: np.ndarray,
        edge: str = "all"
    ) -> Dict[str, np.ndarray]:
        """
        Extract boundary values for sharing with adjacent tiles.
        
        Args:
            heightmap: Input heightmap
            edge: Which edge(s) to extract ("north", "south", "east", "west", "all")
            
        Returns:
            Dictionary mapping edge names to boundary value arrays
        """
        
        boundaries = {}
        
        if edge in ["north", "all"]:
            boundaries["north"] = heightmap[0, :].copy()
        
        if edge in ["south", "all"]:
            boundaries["south"] = heightmap[-1, :].copy()
            
        if edge in ["east", "all"]:
            boundaries["east"] = heightmap[:, -1].copy()
            
        if edge in ["west", "all"]:
            boundaries["west"] = heightmap[:, 0].copy()
        
        return boundaries
    
    def create_grid_continuity_info(
        self,
        world_x: int, world_y: int,
        enable_blending: bool = True
    ) -> Dict[str, Any]:
        """
        Create grid continuity configuration for seamless generation.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            enable_blending: Whether to enable edge blending
            
        Returns:
            Grid continuity configuration dictionary
        """
        
        grid_info = {
            "blend_edges": enable_blending,
            "overlap_size": self.overlap,
            "tile_coordinates": (world_x // self.tile_size, world_y // self.tile_size)
        }
        
        # Add boundary constraints if we have cached values
        cache_key = (world_x // self.tile_size, world_y // self.tile_size)
        if cache_key in self.boundary_cache:
            grid_info["boundary_constraints"] = self.boundary_cache[cache_key]
        
        return grid_info