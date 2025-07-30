"""
Erosion simulation for terrain generation.

Implements hydraulic erosion and thermal erosion using JAX.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def hydraulic_erosion(
    heightmap: jnp.ndarray,
    iterations: int = 200,
    rain_amount: float = 0.5,
    evaporation: float = 0.05,
    capacity: float = 0.3,
    deposition: float = 0.1,
    erosion_speed: float = 0.1,
    seed: int = 0
) -> jnp.ndarray:
    """
    Apply hydraulic erosion to a heightmap.
    
    This is a simplified JAX-compatible version of hydraulic erosion.
    For performance, we use a cellular automata approach rather than
    particle simulation.
    
    Args:
        heightmap: Input height map (2D array)
        iterations: Number of erosion iterations
        rain_amount: Amount of water added each iteration
        evaporation: Water evaporation rate
        capacity: Water carrying capacity
        deposition: Sediment deposition rate
        erosion_speed: How fast erosion occurs
        seed: Random seed
        
    Returns:
        Eroded heightmap
    """
    
    height = heightmap.copy()
    water = jnp.zeros_like(height)
    sediment = jnp.zeros_like(height)
    
    key = jax.random.PRNGKey(seed)
    
    for i in range(iterations):
        key, subkey = jax.random.split(key)
        
        # Add rain (with some randomness)
        rain = rain_amount * (0.8 + 0.4 * jax.random.uniform(subkey, height.shape))
        water = water + rain
        
        # Calculate gradients (flow direction)
        grad_x, grad_y = jnp.gradient(height + water)
        
        # Flow water based on gradients
        # Use a simple approach: water flows to lower adjacent cells
        water_flow = water * jnp.sqrt(grad_x**2 + grad_y**2) * 0.1
        water_flow = jnp.clip(water_flow, 0, water * 0.5)  # Don't flow more than half
        
        # Erosion capacity based on water velocity
        velocity = jnp.sqrt(grad_x**2 + grad_y**2)
        max_sediment = capacity * water * velocity
        
        # Erosion: remove material where water can carry more sediment
        excess_capacity = max_sediment - sediment
        erosion_amount = jnp.where(
            excess_capacity > 0,
            jnp.minimum(excess_capacity, erosion_speed * water),
            0
        )
        
        height = height - erosion_amount
        sediment = sediment + erosion_amount
        
        # Deposition: drop sediment where water can't carry it
        excess_sediment = sediment - max_sediment
        deposition_amount = jnp.where(
            excess_sediment > 0,
            excess_sediment * deposition,
            0
        )
        
        height = height + deposition_amount
        sediment = sediment - deposition_amount
        
        # Evaporation
        water = water * (1.0 - evaporation)
        
        # Ensure non-negative values
        water = jnp.maximum(water, 0)
        sediment = jnp.maximum(sediment, 0)
    
    return height


def thermal_erosion(
    heightmap: jnp.ndarray,
    iterations: int = 50,
    talus_angle: float = 0.5,
    erosion_rate: float = 0.1
) -> jnp.ndarray:
    """
    Apply thermal erosion (gravity-based erosion) to a heightmap.
    
    Material slides down slopes that are too steep.
    
    Args:
        heightmap: Input height map
        iterations: Number of erosion iterations
        talus_angle: Maximum stable slope (higher = steeper slopes allowed)
        erosion_rate: How fast material slides
        
    Returns:
        Eroded heightmap
    """
    
    height = heightmap.copy()
    
    for i in range(iterations):
        # Calculate slopes to neighbors
        height_padded = jnp.pad(height, 1, mode='edge')
        
        # Get height differences to 8 neighbors
        diffs = []
        neighbor_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dy, dx in neighbor_offsets:
            neighbor_height = height_padded[1+dy:height.shape[0]+1+dy, 1+dx:height.shape[1]+1+dx]
            diff = height - neighbor_height
            diffs.append(diff)
        
        # Find maximum height difference
        max_diff = jnp.maximum(*diffs)
        
        # Erosion occurs where slope exceeds talus angle
        erosion_mask = max_diff > talus_angle
        erosion_amount = jnp.where(erosion_mask, (max_diff - talus_angle) * erosion_rate, 0)
        
        # Remove material
        height = height - erosion_amount
        
        # Distribute material to lower neighbors
        for j, (dy, dx) in enumerate(neighbor_offsets):
            neighbor_height = height_padded[1+dy:height.shape[0]+1+dy, 1+dx:height.shape[1]+1+dx]
            
            # Add material where this neighbor is lower
            flow_mask = (diffs[j] > talus_angle) & erosion_mask
            flow_amount = jnp.where(flow_mask, erosion_amount / 8.0, 0)  # Distribute evenly
            
            # Apply the flow (this is approximate due to JAX constraints)
            height = height + flow_amount * 0.1  # Small factor to prevent instability
    
    return height


def river_erosion(
    heightmap: jnp.ndarray,
    start_points: jnp.ndarray,
    erosion_strength: float = 0.1,
    river_width: float = 2.0,
    downhill_bias: float = 0.8
) -> jnp.ndarray:
    """
    Carve river channels in the heightmap.
    
    This is a simplified river erosion that traces paths downhill
    from start points and carves channels.
    
    Args:
        heightmap: Input height map
        start_points: Array of (y, x) starting positions for rivers
        erosion_strength: How deep to carve channels
        river_width: Width of river channels (in pixels)
        downhill_bias: How much rivers prefer going downhill vs random walk
        
    Returns:
        Heightmap with river channels
    """
    
    height = heightmap.copy()
    h, w = height.shape
    
    # Create a distance map for each start point
    for start_y, start_x in start_points:
        if not (0 <= start_y < h and 0 <= start_x < w):
            continue
            
        # Trace path downhill
        current_y, current_x = start_y, start_x
        path_height = height[current_y, current_x]
        
        # Simple downhill path tracing
        for step in range(min(h, w)):  # Limit path length
            # Find steepest downhill direction
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = current_y + dy, current_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbors.append((height[ny, nx], ny, nx))
            
            if not neighbors:
                break
                
            # Sort by height (lowest first)
            neighbors.sort()
            lowest_height, next_y, next_x = neighbors[0]
            
            # Stop if we're not going downhill anymore
            if lowest_height >= path_height:
                break
                
            # Carve channel along this segment
            # Create a line between current and next position
            steps = max(abs(next_y - current_y), abs(next_x - current_x))
            if steps > 0:
                for t in jnp.linspace(0, 1, steps + 1):
                    y = int(current_y + t * (next_y - current_y))
                    x = int(current_x + t * (next_x - current_x))
                    
                    # Carve in a circular pattern around the point
                    for dy in range(-int(river_width), int(river_width) + 1):
                        for dx in range(-int(river_width), int(river_width) + 1):
                            py, px = y + dy, x + dx
                            if 0 <= py < h and 0 <= px < w:
                                dist = jnp.sqrt(dy**2 + dx**2)
                                if dist <= river_width:
                                    erosion_factor = (1.0 - dist / river_width) * erosion_strength
                                    height = height.at[py, px].add(-erosion_factor)
            
            # Move to next position
            current_y, current_x = next_y, next_x
            path_height = lowest_height
    
    return height


def combined_erosion(
    heightmap: jnp.ndarray,
    hydraulic_iters: int = 100,
    thermal_iters: int = 25,
    rain_amount: float = 0.3,
    erosion_strength: float = 0.05,
    seed: int = 0
) -> jnp.ndarray:
    """
    Apply combined hydraulic and thermal erosion.
    
    This gives the most realistic results by combining both erosion types.
    
    Args:
        heightmap: Input height map
        hydraulic_iters: Number of hydraulic erosion iterations
        thermal_iters: Number of thermal erosion iterations
        rain_amount: Rain amount for hydraulic erosion
        erosion_strength: Overall erosion strength
        seed: Random seed
        
    Returns:
        Eroded heightmap
    """
    
    height = heightmap.copy()
    
    # Apply thermal erosion first (creates natural slopes)
    height = thermal_erosion(
        height,
        iterations=thermal_iters,
        talus_angle=0.4,
        erosion_rate=erosion_strength * 0.5
    )
    
    # Then apply hydraulic erosion (creates valleys and channels)
    height = hydraulic_erosion(
        height,
        iterations=hydraulic_iters,
        rain_amount=rain_amount,
        evaporation=0.05,
        capacity=0.2,
        erosion_speed=erosion_strength,
        seed=seed
    )
    
    # Final light thermal erosion to smooth sharp edges
    height = thermal_erosion(
        height,
        iterations=thermal_iters // 2,
        talus_angle=0.3,
        erosion_rate=erosion_strength * 0.2
    )
    
    return height