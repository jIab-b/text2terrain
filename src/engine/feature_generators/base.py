"""
Base terrain generator and common utilities.
"""

import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class FeatureGenerator(ABC):
    """Base class for all feature generators."""
    
    @abstractmethod
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Apply this feature to existing heightmap."""
        pass
    
    def enhance(
        self,
        heightmap: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Apply subtle enhancements (for secondary features)."""
        return heightmap


class BaseGenerator(FeatureGenerator):
    """
    Generates base terrain using simple noise.
    
    Provides foundation that other features build upon.
    """
    
    def generate(
        self,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Generate base terrain from scratch."""
        
        frequency = parameters.get("base_frequency", 0.01)
        amplitude = parameters.get("base_amplitude", 0.3)
        
        # Simple base noise
        noise = self._simple_noise(X, Y, seed, frequency)
        return noise * amplitude
    
    def apply(
        self,
        heightmap: np.ndarray,
        X: np.ndarray, Y: np.ndarray,
        parameters: Dict[str, float],
        seed: int
    ) -> np.ndarray:
        """Add base terrain to existing heightmap."""
        
        base = self.generate(X, Y, parameters, seed)
        return heightmap + base
    
    def _simple_noise(
        self,
        X: np.ndarray, Y: np.ndarray,
        seed: int,
        frequency: float
    ) -> np.ndarray:
        """Generate simple 2D noise."""
        
        # Scale coordinates
        x = X * frequency
        y = Y * frequency
        
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