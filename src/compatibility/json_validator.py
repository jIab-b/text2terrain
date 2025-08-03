"""
JSON format validator for terrain generation data.

Ensures generated JSON data is compatible with renderer expectations.
"""

import json
from typing import Dict, Any, List, Optional


class JSONValidator:
    """
    Validates JSON format for terrain generation compatibility.
    
    Ensures data format matches renderer expectations while
    supporting enhanced features.
    """
    
    def __init__(self):
        self.required_legacy_fields = [
            "tile_u", "tile_v", "world_x", "world_y",
            "global_seed", "module_ids", "parameters", "seeds"
        ]
        
        self.required_parameter_fields = [
            "frequency", "octaves", "persistence", "height_scale"
        ]
        
        self.valid_module_ids = list(range(0, 9))  # 0-8 are valid
    
    def validate_legacy_format(self, data: Dict[str, Any], tile_size: int = 256) -> tuple[bool, List[str]]:
        """
        Validate legacy JSON format compatibility.
        
        Args:
            data: JSON data to validate
            tile_size: Actual tile size used by the generator
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        
        errors = []
        
        # Check required fields
        for field in self.required_legacy_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate field types and values
        errors.extend(self._validate_field_types(data))
        errors.extend(self._validate_field_values(data, tile_size))
        errors.extend(self._validate_parameters(data.get("parameters", {})))
        
        return len(errors) == 0, errors
    
    def _validate_field_types(self, data: Dict[str, Any]) -> List[str]:
        """Validate field types match expectations."""
        
        errors = []
        
        # Integer fields
        int_fields = ["tile_u", "tile_v", "world_x", "world_y", "global_seed"]
        for field in int_fields:
            if field in data and not isinstance(data[field], int):
                errors.append(f"Field {field} must be integer, got {type(data[field])}")
        
        # List fields
        if "module_ids" in data and not isinstance(data["module_ids"], list):
            errors.append("module_ids must be a list")
        
        if "seeds" in data and not isinstance(data["seeds"], list):
            errors.append("seeds must be a list")
        
        # Dict field
        if "parameters" in data and not isinstance(data["parameters"], dict):
            errors.append("parameters must be a dictionary")
        
        return errors
    
    def _validate_field_values(self, data: Dict[str, Any], tile_size: int) -> List[str]:
        """Validate field values are reasonable."""
        
        errors = []
        
        # Validate module IDs
        module_ids = data.get("module_ids", [])
        if len(module_ids) == 0:
            errors.append("module_ids cannot be empty")
        elif len(module_ids) > 10:
            errors.append("module_ids list too long (max 10)")
        
        for i, module_id in enumerate(module_ids):
            if not isinstance(module_id, int):
                errors.append(f"module_ids[{i}] must be integer, got {type(module_id)}")
            elif module_id not in self.valid_module_ids:
                errors.append(f"module_ids[{i}] = {module_id} is not valid (must be 0-8)")
        
        # Validate seeds
        seeds = data.get("seeds", [])
        if len(seeds) != len(module_ids):
            errors.append(f"seeds length ({len(seeds)}) must match module_ids length ({len(module_ids)})")
        
        for i, seed in enumerate(seeds):
            if not isinstance(seed, int):
                errors.append(f"seeds[{i}] must be integer, got {type(seed)}")
            elif seed < 0:
                errors.append(f"seeds[{i}] must be non-negative")
        
        # Validate world coordinates are multiples of tile size
        world_x = data.get("world_x", 0)
        world_y = data.get("world_y", 0)
        
        if world_x % tile_size != 0:
            errors.append(f"world_x ({world_x}) must be multiple of tile size (detected: {tile_size})")
        
        if world_y % tile_size != 0:
            errors.append(f"world_y ({world_y}) must be multiple of tile size (detected: {tile_size})")
        
        # Validate tile coordinates match world coordinates
        tile_u = data.get("tile_u", 0)
        tile_v = data.get("tile_v", 0)
        expected_tile_u = world_x // tile_size
        expected_tile_v = world_y // tile_size
        
        if tile_u != expected_tile_u:
            errors.append(f"tile_u ({tile_u}) doesn't match world_x ({world_x}) with tile_size {tile_size}")
        
        if tile_v != expected_tile_v:
            errors.append(f"tile_v ({tile_v}) doesn't match world_y ({world_y}) with tile_size {tile_size}")
        
        return errors
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameter values are reasonable."""
        
        errors = []
        
        # Check required parameters
        for field in self.required_parameter_fields:
            if field not in parameters:
                errors.append(f"Missing required parameter: {field}")
        
        # Validate parameter ranges
        validations = {
            "frequency": (0.0001, 1.0, "frequency"),
            "octaves": (1, 8, "octaves", int),
            "persistence": (0.1, 1.0, "persistence"),
            "lacunarity": (1.0, 5.0, "lacunarity"),
            "height_scale": (1.0, 10000.0, "height_scale"),
            "ridge_sharpness": (0.0, 3.0, "ridge_sharpness"),
            "warp_amplitude": (0.0, 1000.0, "warp_amplitude"),
            "warp_frequency": (0.0001, 0.1, "warp_frequency"),
            "warp_octaves": (1, 8, "warp_octaves", int),
            "erosion_speed": (0.0, 1.0, "erosion_speed"),
            "rain_amount": (0.0, 2.0, "rain_amount"),
            "evaporation": (0.01, 1.0, "evaporation"),
            "capacity": (0.1, 2.0, "capacity"),
            "iterations": (1, 1000, "iterations", int)
        }
        
        for param_name, validation in validations.items():
            if param_name in parameters:
                value = parameters[param_name]
                min_val, max_val = validation[0], validation[1]
                param_display = validation[2]
                expected_type = validation[3] if len(validation) > 3 else float
                
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param_display} must be numeric, got {type(value)}")
                elif expected_type == int and not isinstance(value, int):
                    errors.append(f"Parameter {param_display} must be integer, got {type(value)}")
                elif value < min_val or value > max_val:
                    errors.append(f"Parameter {param_display} = {value} out of range [{min_val}, {max_val}]")
        
        return errors
    
    def validate_enhanced_format(self, data: Dict[str, Any], tile_size: int = 256) -> tuple[bool, List[str]]:
        """
        Validate enhanced JSON format with new features.
        
        Args:
            data: Enhanced JSON data to validate
            tile_size: Actual tile size used by the generator
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        
        errors = []
        
        # First validate legacy compatibility
        legacy_valid, legacy_errors = self.validate_legacy_format(data, tile_size)
        errors.extend(legacy_errors)
        
        # Validate enhanced fields if present
        if "features" in data:
            errors.extend(self._validate_features(data["features"]))
        
        if "terrain_analysis" in data:
            errors.extend(self._validate_terrain_analysis(data["terrain_analysis"]))
        
        if "grid_continuity" in data:
            errors.extend(self._validate_grid_continuity(data["grid_continuity"]))
        
        return len(errors) == 0, errors
    
    def _validate_features(self, features: Dict[str, Any]) -> List[str]:
        """Validate features configuration."""
        
        errors = []
        
        # Check feature structure
        if not isinstance(features, dict):
            errors.append("features must be a dictionary")
            return errors
        
        # Validate feature lists
        for field in ["primary_features", "secondary_features"]:
            if field in features:
                if not isinstance(features[field], list):
                    errors.append(f"features.{field} must be a list")
                else:
                    for i, feature in enumerate(features[field]):
                        if not isinstance(feature, str):
                            errors.append(f"features.{field}[{i}] must be string")
        
        # Validate terrain type
        if "terrain_type" in features:
            valid_types = ["plains", "hills", "mountains", "valleys", "wetlands"]
            if features["terrain_type"] not in valid_types:
                errors.append(f"Invalid terrain_type: {features['terrain_type']}")
        
        # Validate complexity
        if "complexity" in features:
            complexity = features["complexity"]
            if not isinstance(complexity, (int, float)) or complexity < 0 or complexity > 1:
                errors.append("features.complexity must be number in range [0, 1]")
        
        return errors
    
    def _validate_terrain_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        """Validate terrain analysis data."""
        
        errors = []
        
        if not isinstance(analysis, dict):
            errors.append("terrain_analysis must be a dictionary")
            return errors
        
        # Check required sections
        required_sections = ["elevation_stats", "slope_analysis", "feature_detection"]
        for section in required_sections:
            if section not in analysis:
                errors.append(f"terrain_analysis missing section: {section}")
            elif not isinstance(analysis[section], dict):
                errors.append(f"terrain_analysis.{section} must be a dictionary")
        
        return errors
    
    def _validate_grid_continuity(self, grid_info: Dict[str, Any]) -> List[str]:
        """Validate grid continuity configuration."""
        
        errors = []
        
        if not isinstance(grid_info, dict):
            errors.append("grid_continuity must be a dictionary")
            return errors
        
        # Validate blend_edges flag
        if "blend_edges" in grid_info:
            if not isinstance(grid_info["blend_edges"], bool):
                errors.append("grid_continuity.blend_edges must be boolean")
        
        # Validate overlap_size
        if "overlap_size" in grid_info:
            overlap_size = grid_info["overlap_size"]
            if not isinstance(overlap_size, int) or overlap_size < 0 or overlap_size > 64:
                errors.append("grid_continuity.overlap_size must be integer in range [0, 64]")
        
        # Validate boundary_constraints if present
        if "boundary_constraints" in grid_info:
            constraints = grid_info["boundary_constraints"]
            if not isinstance(constraints, dict):
                errors.append("grid_continuity.boundary_constraints must be a dictionary")
            else:
                valid_edges = ["north", "south", "east", "west"]
                for edge, values in constraints.items():
                    if edge not in valid_edges:
                        errors.append(f"Invalid boundary edge: {edge}")
                    elif not isinstance(values, list):
                        errors.append(f"boundary_constraints.{edge} must be a list")
                    else:
                        for i, value in enumerate(values):
                            if not isinstance(value, (int, float)):
                                errors.append(f"boundary_constraints.{edge}[{i}] must be numeric")
        
        return errors
    
    def sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and fix common data issues.
        
        Args:
            data: Input data to sanitize
            
        Returns:
            Sanitized data
        """
        
        result = data.copy()
        
        # Ensure required fields exist with defaults
        if "global_seed" not in result:
            result["global_seed"] = 42
        
        if "module_ids" not in result:
            result["module_ids"] = [0]  # Default to perlin noise
        
        if "parameters" not in result:
            result["parameters"] = {}
        
        # Ensure required parameters exist
        params = result["parameters"]
        param_defaults = {
            "frequency": 0.01,
            "octaves": 4,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "height_scale": 1000.0
        }
        
        for param, default in param_defaults.items():
            if param not in params:
                params[param] = default
        
        # Generate seeds if missing
        if "seeds" not in result or len(result["seeds"]) != len(result["module_ids"]):
            global_seed = result["global_seed"]
            result["seeds"] = [global_seed + i for i in range(len(result["module_ids"]))]
        
        # Calculate tile coordinates if missing
        if "world_x" in result and "world_y" in result:
            # Infer tile size from coordinates
            tile_size = 256
            world_x, world_y = result["world_x"], result["world_y"]
            if world_x != 0 or world_y != 0:
                for size in [64, 128, 256, 512]:
                    if world_x % size == 0 and world_y % size == 0:
                        tile_size = size
                        break
            
            result.setdefault("tile_u", world_x // tile_size)
            result.setdefault("tile_v", world_y // tile_size)
        
        # Clamp parameter values to valid ranges
        clamps = {
            "frequency": (0.0001, 1.0),
            "octaves": (1, 8),
            "persistence": (0.1, 1.0),
            "height_scale": (1.0, 10000.0)
        }
        
        for param, (min_val, max_val) in clamps.items():
            if param in params:
                if param == "octaves":
                    params[param] = max(min_val, min(max_val, int(params[param])))
                else:
                    params[param] = max(min_val, min(max_val, float(params[param])))
        
        return result
    
    def create_minimal_valid_data(
        self,
        world_x: int = 0,
        world_y: int = 0,
        global_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Create minimal valid JSON data for testing.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate  
            global_seed: Global random seed
            
        Returns:
            Minimal valid JSON data
        """
        
        tile_size = 256
        tile_u = world_x // tile_size
        tile_v = world_y // tile_size
        
        return {
            "tile_u": tile_u,
            "tile_v": tile_v,
            "world_x": world_x,
            "world_y": world_y,
            "global_seed": global_seed,
            "module_ids": [0],  # Just perlin noise
            "parameters": {
                "frequency": 0.01,
                "octaves": 4,
                "persistence": 0.5,
                "lacunarity": 2.0,
                "height_scale": 1000.0
            },
            "seeds": [global_seed]
        }