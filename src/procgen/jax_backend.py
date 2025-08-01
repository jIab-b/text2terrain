"""
Basic JAX-accelerated terrain synthesis kernels.
Currently implements Perlin-style value noise + optional domain warp + ridge transform.
All functions are jit-compiled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Dict


@jax.jit
def _hash22(p: jnp.ndarray) -> jnp.ndarray:
    # Simple 2-D hash to pseudo-random value in [0,1]
    p = (p * jnp.array([127.1, 311.7])) % 1.0
    return jax.random.uniform(jax.random.PRNGKey(int(jnp.sum(p * 1e4))), p.shape)


@jax.jit
def _lerp(a, b, t):
    return a + t * (b - a)


@jax.jit
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@jax.jit
def perlin2d(x: jnp.ndarray, y: jnp.ndarray, freq: float, amp: float, seed: int) -> jnp.ndarray:
    """Very small Perlin-style value noise, deterministic per seed."""
    # scale coordinates to create features
    x = x * freq * 256  # Scale [0,1] coords to meaningful range
    y = y * freq * 256
    # Grid cell coordinates
    x0 = jnp.floor(x)
    y0 = jnp.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # Random values at four corners
    def rand(ix, iy):
        # Convert coords to 32-bit ints for hashing
        ix = ix.astype(jnp.uint32)
        iy = iy.astype(jnp.uint32)
        s  = jnp.uint32(seed)
        h  = (ix * jnp.uint32(374761393) + iy * jnp.uint32(668265263) + s * jnp.uint32(2246822519))
        h ^= (h >> 13)
        h *= jnp.uint32(1274126177)
        h ^= (h >> 16)
        return (h.astype(jnp.float32) / 4294967295.0)

    v00 = rand(x0, y0)
    v10 = rand(x1, y0)
    v01 = rand(x0, y1)
    v11 = rand(x1, y1)

    # Local coordinates
    tx = _fade(x - x0)
    ty = _fade(y - y0)

    # Bilinear blend
    a = _lerp(v00, v10, tx)
    b = _lerp(v01, v11, tx)
    return amp * _lerp(a, b, ty)


@jax.jit
def _domain_warp(h: jnp.ndarray, warp_amp: float, warp_freq: float) -> jnp.ndarray:
    # Use the height itself as offset field
    coords = jnp.arange(h.shape[0])
    xx, yy = jnp.meshgrid(coords, coords, indexing="ij")
    dx = h * warp_amp
    xw = (xx + dx) * warp_freq
    yw = (yy + dx) * warp_freq
    xw = jnp.clip(xw, 0, h.shape[0] - 1)
    yw = jnp.clip(yw, 0, h.shape[1] - 1)
    return jax.scipy.ndimage.map_coordinates(h, [xw, yw], order=1, mode="wrap")


@jax.jit
def _ridge(h: jnp.ndarray, sharp: float) -> jnp.ndarray:
    return 1.0 - jnp.abs(h) ** sharp


@jax.jit
def _normalize(h: jnp.ndarray, scale: float) -> jnp.ndarray:
    hmin = h.min()
    hmax = h.max()
    h = (h - hmin) / jnp.maximum(1e-6, hmax - hmin)
    return h * scale


@jax.jit
def generate(params: Dict[str, float]) -> jnp.ndarray:
    # Construct coordinate grid in [0,1] range
    size = 256
    coords = jnp.linspace(0, 1, size)
    xx, yy = jnp.meshgrid(coords, coords, indexing="ij")
    h = perlin2d(xx, yy, params["frequency"], 1.0, params["seed"])

    h = jnp.where(params["warp_amplitude"] > 0.0, 
                  _domain_warp(h, params["warp_amplitude"], params["warp_frequency"]), 
                  h)
    h = jnp.where(params["ridge_sharpness"] > 0.0,
                  _ridge(h, params["ridge_sharpness"]),
                  h)
    h = _normalize(h, params["height_scale"])
    return h
