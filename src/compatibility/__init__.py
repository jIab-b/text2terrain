"""
Compatibility layer for existing JSON format and renderer.

Maintains backward compatibility while enabling new architecture.
"""

from .legacy_adapter import LegacyAdapter
from .json_validator import JSONValidator

__all__ = ["LegacyAdapter", "JSONValidator"]