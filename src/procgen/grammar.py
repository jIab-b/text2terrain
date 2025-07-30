"""
Module registry and parameter specification for terrain generation.

This module defines:
- ParameterSpec: Validation and extraction of parameters
- ModuleRegistry: Registration and lookup of terrain modules
"""

from typing import Dict, Any, Callable, Tuple, Optional, List
import jax.numpy as jnp


class ParameterSpec:
    """
    Specification for module parameters with validation and normalization.
    
    Each parameter has:
    - min_val: Minimum allowed value
    - max_val: Maximum allowed value  
    - default: Default value if not specified
    """
    
    def __init__(self, params: Dict[str, Tuple[float, float, float]]):
        """
        Initialize parameter specification.
        
        Args:
            params: Dict mapping param_name -> (min_val, max_val, default)
        """
        self.params = params
    
    def validate(self, values: Dict[str, float]) -> bool:
        """Check if all required parameters are present and in valid ranges."""
        
        for param_name, (min_val, max_val, _) in self.params.items():
            if param_name not in values:
                return False
            
            value = values[param_name]
            if not (min_val <= value <= max_val):
                return False
        
        return True
    
    def extract_params(self, values: Dict[str, float]) -> Dict[str, float]:
        """Extract and validate parameters for this module."""
        
        result = {}
        for param_name, (min_val, max_val, default) in self.params.items():
            if param_name in values:
                value = values[param_name]
                # Clamp to valid range
                value = max(min_val, min(max_val, value))
                result[param_name] = value
            else:
                result[param_name] = default
        
        return result
    
    def normalize_params(self, values: Dict[str, float]) -> Dict[str, float]:
        """Normalize parameters to [0, 1] range for neural network training."""
        
        result = {}
        for param_name, (min_val, max_val, _) in self.params.items():
            if param_name in values:
                value = values[param_name]
                # Normalize to [0, 1]
                normalized = (value - min_val) / (max_val - min_val)
                result[param_name] = normalized
        
        return result
    
    def denormalize_params(self, normalized_values: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized [0, 1] parameters back to original ranges."""
        
        result = {}
        for param_name, (min_val, max_val, default) in self.params.items():
            if param_name in normalized_values:
                normalized = normalized_values[param_name]
                # Clamp normalized value to [0, 1]
                normalized = max(0.0, min(1.0, normalized))
                # Denormalize
                value = min_val + normalized * (max_val - min_val)
                result[param_name] = value
            else:
                result[param_name] = default
        
        return result
    
    def get_param_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.params.keys())
    
    def get_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges (min, max) for each parameter."""
        return {name: (min_val, max_val) for name, (min_val, max_val, _) in self.params.items()}


class ModuleRegistry:
    """
    Registry for terrain generation modules.
    
    Manages module registration, lookup, and parameter specifications.
    """
    
    def __init__(self):
        self.modules: Dict[str, Callable] = {}
        self.param_specs: Dict[str, ParameterSpec] = {}
        self.name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}
        self._next_id = 0
    
    def register(self, name: str, func: Callable, param_spec: ParameterSpec):
        """Register a new terrain module."""
        
        module_id = self._next_id
        self._next_id += 1
        
        self.modules[name] = func
        self.param_specs[name] = param_spec
        self.name_to_id[name] = module_id
        self.id_to_name[module_id] = name
    
    def get_module_function(self, module_id: int) -> Callable:
        """Get module function by ID."""
        name = self.id_to_name[module_id]
        return self.modules[name]
    
    def get_module_name(self, module_id: int) -> str:
        """Get module name by ID."""
        return self.id_to_name[module_id]
    
    def get_module_id(self, name: str) -> int:
        """Get module ID by name."""
        return self.name_to_id[name]
    
    def get_parameter_spec(self, module_id: int) -> ParameterSpec:
        """Get parameter specification by module ID."""
        name = self.id_to_name[module_id]
        return self.param_specs[name]
    
    def list_modules(self) -> List[Tuple[int, str]]:
        """List all registered modules as (id, name) pairs."""
        return [(module_id, name) for name, module_id in self.name_to_id.items()]
    
    def get_all_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all parameters from all modules (for training data generation)."""
        
        all_params = {}
        for param_spec in self.param_specs.values():
            all_params.update(param_spec.params)
        
        return all_params