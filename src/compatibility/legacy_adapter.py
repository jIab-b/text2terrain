"""
Legacy format adapter for backward compatibility.

Converts between new feature-based format and legacy parameter format
to maintain compatibility with existing renderer.
"""

import json
from typing import Dict, List, Any, Tuple
import numpy as np


class LegacyAdapter:
    """
    Converts between new feature-based format and legacy JSON format.
    
    Ensures existing renderer continues to work while enabling
    new feature-based terrain generation.
    """
    
    # Legacy module ID mapping
    LEGACY_MODULE_MAP = {
        "perlin_noise": 0,
        "ridged_multi": 1,
        "domain_warp": 2, 
        "hydraulic_erosion": 3,
        # New feature generators
        "mountains": 4,
        "valleys": 5,
        "caves": 6,
        "rivers": 7,
        "biomes": 8
    }
    
    # Feature to module mapping
    FEATURE_TO_MODULE_MAP = {
        "steep_peaks": [0, 1, 4],  # perlin + ridged + mountains
        "mountain_peaks": [1, 4],  # ridged + mountains
        "rolling_hills": [0, 4],   # perlin + mountains (gentle)
        "deep_valleys": [0, 5],    # perlin + valleys
        "river_valleys": [5, 7],   # valleys + rivers
        "cave_systems": [6],       # caves
        "water_features": [7],     # rivers
        "erosion": [3],           # hydraulic_erosion
        "ridges": [1],            # ridged_multi
        "plains": [0],            # perlin_noise
        "desert": [0, 8],         # perlin + biomes
        "forest": [0, 8],         # perlin + biomes
        "arctic": [0, 8],         # perlin + biomes
        "volcanic": [1, 8],       # ridged + biomes
        "swamp": [0, 7, 8]        # perlin + rivers + biomes
    }
    
    def __init__(self):
        self.reverse_module_map = {v: k for k, v in self.LEGACY_MODULE_MAP.items()}
    
    def convert_to_legacy_format(
        self,
        features: Dict[str, Any],
        parameters: Dict[str, float],
        world_x: int,
        world_y: int,
        seeds: List[int],
        global_seed: int = 42,
        tile_size: int = 256
    ) -> Dict[str, Any]:
        """
        Convert new feature-based format to legacy JSON format.
        
        Args:
            features: New feature configuration
            parameters: Feature parameters
            world_x: World X coordinate
            world_y: World Y coordinate
            seeds: Random seeds
            global_seed: Global random seed
            
        Returns:
            Legacy format dictionary
        """
        
        # Map features to legacy module IDs
        module_ids = self._map_features_to_modules(features)
        
        # Convert feature parameters to legacy parameter names
        legacy_params = self._map_parameters_to_legacy(parameters, features)
        
        # Calculate tile coordinates
        tile_u = world_x // tile_size
        tile_v = world_y // tile_size
        
        # Ensure we have enough seeds
        while len(seeds) < len(module_ids):
            seeds.append(global_seed + len(seeds))
        
        return {
            "tile_u": tile_u,
            "tile_v": tile_v,
            "world_x": world_x,
            "world_y": world_y,
            "global_seed": global_seed,
            "module_ids": module_ids[:10],  # Limit to reasonable number
            "parameters": legacy_params,
            "seeds": seeds[:len(module_ids)]
        }
    
    def _map_features_to_modules(self, features: Dict[str, Any]) -> List[int]:
        """Map feature names to legacy module IDs."""
        
        module_ids = []
        
        # Always include base terrain
        module_ids.append(self.LEGACY_MODULE_MAP["perlin_noise"])
        
        # Map primary features
        for feature_name in features.get("primary_features", []):
            if feature_name in self.FEATURE_TO_MODULE_MAP:
                for module_name in self.FEATURE_TO_MODULE_MAP[feature_name]:
                    if isinstance(module_name, str):
                        if module_name in self.LEGACY_MODULE_MAP:
                            module_id = self.LEGACY_MODULE_MAP[module_name]
                            if module_id not in module_ids:
                                module_ids.append(module_id)
                    else:
                        # Direct module ID
                        if module_name not in module_ids:
                            module_ids.append(module_name)
        
        # Map secondary features
        for feature_name in features.get("secondary_features", []):
            if feature_name in self.FEATURE_TO_MODULE_MAP:
                for module_name in self.FEATURE_TO_MODULE_MAP[feature_name]:
                    if isinstance(module_name, str):
                        if module_name in self.LEGACY_MODULE_MAP:
                            module_id = self.LEGACY_MODULE_MAP[module_name]
                            if module_id not in module_ids:
                                module_ids.append(module_id)
                    else:
                        # Direct module ID
                        if module_name not in module_ids:
                            module_ids.append(module_name)
        
        # Add biome if specified
        if "biome" in features and features["biome"] != "plains":
            biome_module_id = self.LEGACY_MODULE_MAP["biomes"]
            if biome_module_id not in module_ids:
                module_ids.append(biome_module_id)
        
        return module_ids
    
    def _map_parameters_to_legacy(
        self,
        parameters: Dict[str, float],
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Map feature parameters to legacy parameter names."""
        
        legacy_params = {}
        
        # Always include required legacy parameters with defaults
        legacy_params.update({
            "frequency": parameters.get("base_frequency", parameters.get("frequency", 0.01)),
            "octaves": int(parameters.get("detail_octaves", parameters.get("octaves", 4))),
            "persistence": parameters.get("detail_persistence", parameters.get("persistence", 0.5)),
            "lacunarity": parameters.get("lacunarity", 2.0),
            "height_scale": parameters.get("height_scale", 1000.0)
        })
        
        # Map feature-specific parameters to legacy equivalents
        
        # Mountain parameters
        if "mountain_height" in parameters:
            legacy_params["ridge_sharpness"] = parameters.get("mountain_steepness", 0.6)
            # Adjust frequency for mountain scale
            if parameters["mountain_height"] > 0.6:
                legacy_params["frequency"] = max(0.005, legacy_params["frequency"] * 0.5)
        
        # Valley parameters
        if "valley_depth" in parameters:
            legacy_params["erosion_speed"] = parameters["valley_depth"] * 0.3
            legacy_params["rain_amount"] = parameters.get("valley_width", 0.4) * 0.8
        
        # Cave parameters (mapped to erosion for legacy compatibility)
        if "cave_density" in parameters:
            legacy_params["erosion_speed"] = max(
                legacy_params.get("erosion_speed", 0),
                parameters["cave_density"] * 0.2
            )
            legacy_params["iterations"] = int(parameters.get("cave_size", 0.5) * 100) + 50
        
        # River parameters
        if "river_depth" in parameters:
            legacy_params["erosion_speed"] = max(
                legacy_params.get("erosion_speed", 0),
                parameters["river_depth"] * 0.4
            )
            legacy_params["rain_amount"] = parameters.get("river_width", 0.3) * 1.2
        
        # Domain warping parameters
        if "warp_amplitude" in parameters:
            legacy_params["warp_amplitude"] = parameters["warp_amplitude"]
            legacy_params["warp_frequency"] = parameters.get("warp_frequency", 0.005)
            legacy_params["warp_octaves"] = int(parameters.get("warp_octaves", 2))
        else:
            # Add some warping if terrain has complex features
            complexity = features.get("complexity", 0.5)
            if complexity > 0.6:
                legacy_params["warp_amplitude"] = complexity * 200
                legacy_params["warp_frequency"] = 0.005
                legacy_params["warp_octaves"] = 2
        
        # Biome-specific adjustments
        biome = features.get("biome", "plains")
        if biome == "desert":
            legacy_params["frequency"] *= 1.5  # More detailed for dunes
            legacy_params["erosion_speed"] = legacy_params.get("erosion_speed", 0) * 0.5  # Less erosion
        elif biome == "arctic":
            legacy_params["frequency"] *= 0.7  # Smoother terrain
            legacy_params["persistence"] *= 0.8  # Less detail
        elif biome == "volcanic":
            legacy_params["ridge_sharpness"] = max(legacy_params.get("ridge_sharpness", 0), 1.0)
            legacy_params["frequency"] *= 1.2
        
        # Ensure all erosion parameters are present if erosion is used
        if legacy_params.get("erosion_speed", 0) > 0:
            legacy_params.setdefault("rain_amount", 0.5)
            legacy_params.setdefault("evaporation", 0.05)
            legacy_params.setdefault("capacity", 0.3)
            legacy_params.setdefault("iterations", 100)
        
        return legacy_params
    
    def convert_from_legacy_format(
        self,
        legacy_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Convert legacy format to new feature-based format.
        
        Args:
            legacy_data: Legacy format dictionary
            
        Returns:
            Tuple of (features, parameters)
        """
        
        module_ids = legacy_data.get("module_ids", [])
        parameters = legacy_data.get("parameters", {})
        
        # Determine features from module IDs
        features = self._infer_features_from_modules(module_ids, parameters)
        
        # Convert legacy parameters to feature parameters
        feature_params = self._map_legacy_parameters(parameters, features)
        
        return features, feature_params
    
    def _infer_features_from_modules(
        self,
        module_ids: List[int],
        parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """Infer features from legacy module IDs and parameters."""
        
        primary_features = []
        secondary_features = []
        terrain_type = "plains"
        biome = "plains"
        
        # Analyze module combination
        has_ridged = 1 in module_ids
        has_erosion = 3 in module_ids
        has_warp = 2 in module_ids
        has_mountains = 4 in module_ids
        has_valleys = 5 in module_ids
        has_caves = 6 in module_ids
        has_rivers = 7 in module_ids
        has_biomes = 8 in module_ids
        
        # Infer primary features
        if has_mountains or (has_ridged and parameters.get("ridge_sharpness", 0) > 0.5):
            if parameters.get("ridge_sharpness", 0) > 0.8:
                primary_features.append("steep_peaks")
                terrain_type = "mountains"
            else:
                primary_features.append("mountain_peaks")
                terrain_type = "mountains"
        
        if has_valleys or (has_erosion and parameters.get("erosion_speed", 0) > 0.2):
            primary_features.append("deep_valleys")
            if terrain_type == "plains":
                terrain_type = "valleys"
        
        if has_caves:
            primary_features.append("cave_systems")
        
        if has_rivers:
            primary_features.append("water_features")
        
        # Infer secondary features
        if has_erosion:
            secondary_features.append("erosion")
        
        if has_warp:
            secondary_features.append("warping")
        
        if has_ridged and "mountain_peaks" not in primary_features:
            secondary_features.append("ridges")
        
        # Infer biome
        if has_biomes:
            # Use parameter hints to guess biome
            if parameters.get("erosion_speed", 0) < 0.1:
                biome = "desert"
            elif parameters.get("rain_amount", 0.5) > 0.8:
                biome = "swamp"
            elif parameters.get("ridge_sharpness", 0) > 0.8:
                biome = "volcanic"
            else:
                biome = "forest"
        
        return {
            "terrain_type": terrain_type,
            "primary_features": primary_features,
            "secondary_features": secondary_features,
            "biome": biome,
            "complexity": min(1.0, len(primary_features) * 0.3 + len(secondary_features) * 0.2)
        }
    
    def _map_legacy_parameters(
        self,
        legacy_params: Dict[str, float],
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Map legacy parameters to feature parameters."""
        
        feature_params = {}
        
        # Basic terrain parameters
        feature_params["base_frequency"] = legacy_params.get("frequency", 0.01)
        feature_params["detail_octaves"] = legacy_params.get("octaves", 4)
        feature_params["detail_persistence"] = legacy_params.get("persistence", 0.5)
        feature_params["height_scale"] = legacy_params.get("height_scale", 1000.0)
        
        # Mountain parameters
        if "mountain_peaks" in features.get("primary_features", []) or \
           "steep_peaks" in features.get("primary_features", []):
            feature_params["mountain_height"] = min(1.0, legacy_params.get("ridge_sharpness", 0.5) + 0.3)
            feature_params["mountain_steepness"] = legacy_params.get("ridge_sharpness", 0.5)
            feature_params["peak_count"] = 3
        
        # Valley parameters
        if "deep_valleys" in features.get("primary_features", []):
            feature_params["valley_depth"] = legacy_params.get("erosion_speed", 0.1) * 3.0
            feature_params["valley_width"] = legacy_params.get("rain_amount", 0.5) * 0.6
            feature_params["valley_count"] = 2
        
        # Cave parameters
        if "cave_systems" in features.get("primary_features", []):
            feature_params["cave_density"] = legacy_params.get("erosion_speed", 0.1) * 2.0
            feature_params["cave_size"] = (legacy_params.get("iterations", 100) - 50) / 100.0
            feature_params["tunnel_width"] = 0.3
        
        # River parameters
        if "water_features" in features.get("primary_features", []):
            feature_params["river_depth"] = legacy_params.get("erosion_speed", 0.1) * 1.5
            feature_params["river_width"] = legacy_params.get("rain_amount", 0.5) * 0.4
            feature_params["river_count"] = 2
        
        # Warping parameters
        if legacy_params.get("warp_amplitude", 0) > 0:
            feature_params["warp_amplitude"] = legacy_params["warp_amplitude"]
            feature_params["warp_frequency"] = legacy_params.get("warp_frequency", 0.005)
        
        # Biome parameters
        biome = features.get("biome", "plains")
        if biome != "plains":
            feature_params["biome_type"] = biome
            feature_params["biome_strength"] = 0.6
        
        return feature_params
    
    def create_enhanced_json(
        self,
        legacy_data: Dict[str, Any],
        features: Dict[str, Any],
        terrain_analysis: Dict[str, Any],
        grid_continuity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create enhanced JSON format with backward compatibility.
        
        Args:
            legacy_data: Legacy format data
            features: New feature configuration
            terrain_analysis: Heightmap analysis results
            grid_continuity: Grid continuity information
            
        Returns:
            Enhanced JSON maintaining legacy compatibility
        """
        
        return {
            # Legacy fields (required for renderer compatibility)
            **legacy_data,
            
            # New enhancement fields (ignored by legacy renderer)
            "features": features,
            "terrain_analysis": terrain_analysis,
            "grid_continuity": grid_continuity
        }
    
    def validate_legacy_compatibility(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate that JSON data is compatible with legacy renderer.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            True if compatible with legacy renderer
        """
        
        required_fields = [
            "tile_u", "tile_v", "world_x", "world_y",
            "global_seed", "module_ids", "parameters", "seeds"
        ]
        
        # Check all required fields exist
        for field in required_fields:
            if field not in json_data:
                return False
        
        # Validate field types
        if not isinstance(json_data["module_ids"], list):
            return False
        
        if not isinstance(json_data["parameters"], dict):
            return False
        
        if not isinstance(json_data["seeds"], list):
            return False
        
        # Validate module IDs are reasonable
        module_ids = json_data["module_ids"]
        if len(module_ids) == 0 or len(module_ids) > 10:
            return False
        
        # Validate seeds match module count
        seeds = json_data["seeds"]
        if len(seeds) != len(module_ids):
            return False
        
        return True