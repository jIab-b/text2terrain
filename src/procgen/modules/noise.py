"""
Noise functions for terrain generation.

JAX-based implementations of various noise algorithms:
- Perlin noise
- Ridged multifractal noise
- Worley noise (cellular)
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def hash_coord(x: jnp.ndarray, y: jnp.ndarray, seed: int = 0) -> jnp.ndarray:
    """Simple hash function for coordinates."""
    seed_arr = jnp.array(seed, dtype=x.dtype)
    h = (x * 374761393 + y * 668265263 + seed_arr * 1664525) % 2147483647
    return (h / 2147483647.0) * 2.0 - 1.0


def smooth_step(t: jnp.ndarray) -> jnp.ndarray:
    """Smooth interpolation function (3t² - 2t³)."""
    return t * t * (3.0 - 2.0 * t)


def perlin_noise(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    frequency: float = 0.01,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0
) -> jnp.ndarray:
    """
    Generate Perlin noise.
    
    Args:
        x, y: Coordinate arrays
        frequency: Base frequency of the noise
        octaves: Number of octaves to sum
        persistence: Amplitude reduction per octave
        lacunarity: Frequency multiplication per octave
        seed: Random seed
        
    Returns:
        Noise values in range approximately [-1, 1]
    """
    
    total = jnp.zeros_like(x)
    amplitude = 1.0
    freq = frequency
    max_value = 0.0
    
    for i in range(octaves):
        # Scale coordinates by frequency
        sx = x * freq
        sy = y * freq
        
        # Get grid cell coordinates
        x0 = jnp.floor(sx).astype(jnp.int64)
        y0 = jnp.floor(sy).astype(jnp.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Get fractional part
        fx = sx - x0
        fy = sy - y0
        
        # Generate gradients at corners
        g00 = hash_coord(x0, y0, seed + i)
        g10 = hash_coord(x1, y0, seed + i)
        g01 = hash_coord(x0, y1, seed + i)
        g11 = hash_coord(x1, y1, seed + i)
        
        # Compute dot products
        d00 = g00 * fx + g00 * fy
        d10 = g10 * (fx - 1) + g10 * fy
        d01 = g01 * fx + g01 * (fy - 1)
        d11 = g11 * (fx - 1) + g11 * (fy - 1)
        
        # Interpolate
        u = smooth_step(fx)
        v = smooth_step(fy)

        nx0 = d00 + u * (d10 - d00)
        nx1 = d01 + u * (d11 - d01)
        noise_val = nx0 + v * (nx1 - nx0)
        
        total += noise_val * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        freq *= lacunarity
    
    return total / max_value


def ridged_multifractal(
    x: jnp.ndarray,
    y: jnp.ndarray,
    frequency: float = 0.01,
    octaves: int = 4,
    persistence: float = 0.5,
    ridge_sharpness: float = 1.0,
    seed: int = 0
) -> jnp.ndarray:
    """
    Generate ridged multifractal noise (good for mountain ridges).
    
    Args:
        x, y: Coordinate arrays
        frequency: Base frequency
        octaves: Number of octaves
        persistence: Amplitude reduction per octave
        ridge_sharpness: How sharp the ridges are (higher = sharper)
        seed: Random seed
        
    Returns:
        Noise values
    """
    
    total = jnp.zeros_like(x)
    amplitude = 1.0
    freq = frequency
    
    for i in range(octaves):
        # Get basic noise
        noise_val = perlin_noise(x, y, freq, 1, 1.0, 1.0, seed + i)
        
        # Create ridges by taking absolute value and inverting
        noise_val = jnp.abs(noise_val)
        noise_val = 1.0 - noise_val
        
        # Sharp ridges
        noise_val = jnp.power(noise_val, ridge_sharpness)
        
        total += noise_val * amplitude
        amplitude *= persistence
        freq *= 2.0
    
    return total


def worley_noise(
    x: jnp.ndarray,
    y: jnp.ndarray, 
    frequency: float = 0.01,
    seed: int = 0
) -> jnp.ndarray:
    """
    Generate Worley (cellular) noise.
    
    Creates cell-like patterns useful for crater fields, cellular structures.
    
    Args:
        x, y: Coordinate arrays
        frequency: Cell density (higher = smaller cells)
        seed: Random seed
        
    Returns:
        Distance to nearest cell center
    """
    
    # Scale coordinates
    sx = x * frequency
    sy = y * frequency
    
    # Get grid cell
    cell_x = jnp.floor(sx).astype(jnp.int64)
    cell_y = jnp.floor(sy).astype(jnp.int64)
    
    min_dist = jnp.full_like(x, jnp.inf)
    
    # Check neighboring cells
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Neighbor cell
            nx = cell_x + dx
            ny = cell_y + dy
            
            # Random point in neighbor cell
            rand_x = hash_coord(nx, ny, seed) * 0.5 + 0.5
            rand_y = hash_coord(nx, ny, seed + 1) * 0.5 + 0.5
            
            # Point position
            point_x = nx + rand_x
            point_y = ny + rand_y
            
            # Distance to point
            dist = jnp.sqrt((sx - point_x)**2 + (sy - point_y)**2)
            min_dist = jnp.minimum(min_dist, dist)
    
    return min_dist


def fbm_noise(
    x: jnp.ndarray,
    y: jnp.ndarray,
    frequency: float = 0.01,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0
) -> jnp.ndarray:
    """
    Generate fractional Brownian motion (fBm) noise.
    
    This is essentially multi-octave Perlin noise with standard parameters.
    """
    return perlin_noise(x, y, frequency, octaves, persistence, lacunarity, seed)