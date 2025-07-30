"""
Domain warping functions for terrain generation.

Domain warping distorts the coordinate space before applying noise,
creating more organic and varied terrain patterns.
"""

import jax.numpy as jnp
from typing import Tuple
from .noise import perlin_noise


def domain_warp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    warp_amplitude: float = 100.0,
    warp_frequency: float = 0.005,
    warp_octaves: int = 2,
    seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply domain warping to coordinate arrays.
    
    Args:
        x, y: Input coordinate arrays
        warp_amplitude: How much to distort coordinates
        warp_frequency: Frequency of the warp noise
        warp_octaves: Number of octaves for warp noise
        seed: Random seed
        
    Returns:
        Tuple of (warped_x, warped_y) coordinate arrays
    """
    
    # Generate warp offsets using noise
    warp_x = perlin_noise(
        x, y,
        frequency=warp_frequency,
        octaves=warp_octaves,
        persistence=0.5,
        lacunarity=2.0,
        seed=seed
    ) * warp_amplitude
    
    warp_y = perlin_noise(
        x, y,
        frequency=warp_frequency,
        octaves=warp_octaves,
        persistence=0.5,
        lacunarity=2.0,
        seed=seed + 1000  # Different seed for Y warp
    ) * warp_amplitude
    
    # Apply warping
    warped_x = x + warp_x
    warped_y = y + warp_y
    
    return warped_x, warped_y


def spiral_warp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
    spiral_strength: float = 0.001,
    spiral_tightness: float = 1.0,
    seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply spiral/rotational warping around a center point.
    
    Creates swirling, hurricane-like patterns.
    
    Args:
        x, y: Input coordinate arrays
        center_x, center_y: Center of spiral
        spiral_strength: How strong the spiral effect is
        spiral_tightness: How tight the spiral is
        seed: Random seed (unused but kept for consistency)
        
    Returns:
        Tuple of (warped_x, warped_y)
    """
    
    # Distance from center
    dx = x - center_x
    dy = y - center_y
    dist = jnp.sqrt(dx**2 + dy**2)
    
    # Angle from center
    angle = jnp.arctan2(dy, dx)
    
    # Add spiral rotation based on distance
    spiral_angle = dist * spiral_strength * spiral_tightness
    new_angle = angle + spiral_angle
    
    # Convert back to coordinates
    warped_x = center_x + dist * jnp.cos(new_angle)
    warped_y = center_y + dist * jnp.sin(new_angle)
    
    return warped_x, warped_y


def radial_warp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
    warp_strength: float = 0.1,
    inner_radius: float = 100.0,
    outer_radius: float = 500.0,
    seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply radial warping (expansion/contraction) around a center.
    
    Creates crater-like or volcanic formations.
    
    Args:
        x, y: Input coordinate arrays
        center_x, center_y: Center of radial warp
        warp_strength: Strength of expansion (+) or contraction (-)
        inner_radius: Start of warp effect
        outer_radius: End of warp effect
        seed: Random seed (unused)
        
    Returns:
        Tuple of (warped_x, warped_y)
    """
    
    # Distance from center
    dx = x - center_x
    dy = y - center_y
    dist = jnp.sqrt(dx**2 + dy**2)
    
    # Normalized direction
    norm_dx = jnp.where(dist > 0, dx / dist, 0)
    norm_dy = jnp.where(dist > 0, dy / dist, 0)
    
    # Warp factor (smooth transition between inner and outer radius)
    warp_factor = jnp.clip(
        (dist - inner_radius) / (outer_radius - inner_radius),
        0.0, 1.0
    )
    
    # Smooth the transition
    warp_factor = warp_factor * warp_factor * (3.0 - 2.0 * warp_factor)
    
    # Apply radial warp
    warp_amount = warp_strength * warp_factor * dist
    warped_x = x + norm_dx * warp_amount
    warped_y = y + norm_dy * warp_amount
    
    return warped_x, warped_y


def multi_warp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    warp_amplitude: float = 100.0,
    warp_frequency: float = 0.005,
    warp_octaves: int = 3,
    recursive_depth: int = 2,
    seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply multiple layers of domain warping recursively.
    
    Creates very complex, organic terrain patterns.
    
    Args:
        x, y: Input coordinate arrays
        warp_amplitude: Base warp amplitude
        warp_frequency: Base warp frequency
        warp_octaves: Octaves per warp layer
        recursive_depth: Number of warp layers to apply
        seed: Random seed
        
    Returns:
        Tuple of (warped_x, warped_y)
    """
    
    warped_x, warped_y = x, y
    
    for i in range(recursive_depth):
        # Each layer uses different frequency and amplitude
        layer_freq = warp_frequency * (2.0 ** i)
        layer_amp = warp_amplitude / (1.5 ** i)
        layer_seed = seed + i * 1000
        
        warped_x, warped_y = domain_warp(
            warped_x, warped_y,
            warp_amplitude=layer_amp,
            warp_frequency=layer_freq,
            warp_octaves=warp_octaves,
            seed=layer_seed
        )
    
    return warped_x, warped_y