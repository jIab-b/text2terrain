"""
Feature-based caption generation from terrain analysis.

Generates natural language descriptions based on actual terrain features
detected from heightmaps rather than parameters.
"""

import random
from typing import Dict, List, Any
import numpy as np


class FeatureCaptionGenerator:
    """
    Generates captions from analyzed terrain features.
    
    Creates natural language descriptions based on actual detected
    terrain characteristics rather than generation parameters.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._setup_vocabulary()
    
    def _setup_vocabulary(self):
        """Initialize vocabulary and templates for caption generation."""
        
        # Terrain type descriptors
        self.terrain_descriptors = {
            "mountains": {
                "adjectives": ["towering", "majestic", "steep", "rugged", "dramatic", "imposing"],
                "nouns": ["peaks", "summits", "ridges", "slopes", "crests", "massifs"]
            },
            "hills": {
                "adjectives": ["rolling", "gentle", "undulating", "soft", "rounded", "smooth"],
                "nouns": ["hills", "mounds", "knolls", "rises", "elevations"]
            },
            "valleys": {
                "adjectives": ["deep", "narrow", "winding", "carved", "sheltered", "secluded"],
                "nouns": ["valleys", "gorges", "ravines", "canyons", "basins", "hollows"]
            },
            "plains": {
                "adjectives": ["vast", "open", "flat", "expansive", "broad", "sweeping"],
                "nouns": ["plains", "flats", "fields", "meadows", "steppes", "prairies"]
            },
            "wetlands": {
                "adjectives": ["marshy", "boggy", "waterlogged", "muddy", "swampy"],
                "nouns": ["wetlands", "marshes", "swamps", "bogs", "fens"]
            }
        }
        
        # Feature-specific descriptors
        self.feature_descriptors = {
            "peaks": ["sharp", "pointed", "jagged", "needle-like", "spire-like"],
            "ridges": ["knife-edge", "serrated", "craggy", "exposed", "windswept"],
            "caves": ["hidden", "mysterious", "underground", "cavernous", "hollow"],
            "water": ["flowing", "meandering", "crystal-clear", "rushing", "still"],
            "erosion": ["weathered", "carved", "sculpted", "worn", "ancient"],
            "slopes": ["steep", "gentle", "gradual", "precipitous", "terraced"]
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            "low": ["subtle", "slight", "mild", "gentle", "soft"],
            "medium": ["moderate", "noticeable", "distinct", "clear"],
            "high": ["dramatic", "extreme", "intense", "pronounced", "striking"]
        }
        
        # Environmental descriptors
        self.environments = [
            "alpine", "arctic", "temperate", "tropical", "arid", "coastal",
            "forested", "barren", "lush", "windswept", "sunlit", "shadowed"
        ]
        
        # Geological terms
        self.geological_terms = [
            "granite", "limestone", "sandstone", "volcanic", "sedimentary",
            "rocky", "boulder-strewn", "scree-covered", "cliff-bound"
        ]
    
    def generate_from_analysis(
        self,
        terrain_analysis: Dict[str, Any],
        features: Dict[str, Any],
        parameters: Dict[str, float]
    ) -> str:
        """
        Generate caption from terrain analysis results.
        
        Args:
            terrain_analysis: Results from HeightmapAnalyzer
            features: Feature configuration used for generation
            parameters: Generation parameters
            
        Returns:
            Natural language caption describing the terrain
        """
        
        # Extract key information
        elevation_stats = terrain_analysis.get("elevation_stats", {})
        slope_analysis = terrain_analysis.get("slope_analysis", {})
        feature_detection = terrain_analysis.get("feature_detection", {})
        terrain_classification = terrain_analysis.get("terrain_classification", {})
        
        # Build caption components
        components = []
        
        # Add environmental descriptor (30% chance)
        if self.rng.random() < 0.3:
            components.append(self.rng.choice(self.environments))
        
        # Add geological descriptor (25% chance)
        if self.rng.random() < 0.25:
            components.append(self.rng.choice(self.geological_terms))
        
        # Primary terrain description
        primary_terrain = self._describe_primary_terrain(
            terrain_classification, elevation_stats, slope_analysis
        )
        components.extend(primary_terrain)
        
        # Feature-specific descriptions
        feature_descriptions = self._describe_features(feature_detection, slope_analysis)
        components.extend(feature_descriptions)
        
        # Add complexity indicators
        complexity_desc = self._describe_complexity(terrain_analysis, features)
        if complexity_desc:
            components.extend(complexity_desc)
        
        # Join components naturally
        return self._join_components(components)
    
    def _describe_primary_terrain(
        self,
        classification: Dict[str, Any],
        elevation_stats: Dict[str, float],
        slope_analysis: Dict[str, float]
    ) -> List[str]:
        """Describe the primary terrain type."""
        
        terrain_type = classification.get("primary_type", "plains")
        confidence = classification.get("confidence", 0.5)
        
        if terrain_type not in self.terrain_descriptors:
            terrain_type = "plains"
        
        descriptors = self.terrain_descriptors[terrain_type]
        
        # Choose intensity based on terrain characteristics
        elevation_range = elevation_stats.get("range", 0)
        mean_slope = slope_analysis.get("mean_slope", 0)
        
        if elevation_range > elevation_stats.get("std", 0) * 2 and mean_slope > 0.3:
            intensity = "high"
        elif elevation_range > elevation_stats.get("std", 0) and mean_slope > 0.1:
            intensity = "medium"
        else:
            intensity = "low"
        
        # Build description
        components = []
        
        # Add intensity modifier if terrain is pronounced
        if intensity != "low" and confidence > 0.6:
            modifier = self.rng.choice(self.intensity_modifiers[intensity])
            components.append(modifier)
        
        # Add terrain adjective
        adjective = self.rng.choice(descriptors["adjectives"])
        components.append(adjective)
        
        # Add terrain noun
        noun = self.rng.choice(descriptors["nouns"])
        components.append(noun)
        
        return components
    
    def _describe_features(
        self,
        feature_detection: Dict[str, Any],
        slope_analysis: Dict[str, float]
    ) -> List[str]:
        """Describe specific detected features."""
        
        descriptions = []
        
        # Describe peaks
        peaks_detected = feature_detection.get("peaks_detected", 0)
        if peaks_detected > 0:
            if peaks_detected == 1:
                descriptions.append("with a prominent peak")
            elif peaks_detected <= 3:
                descriptions.append("with several peaks")
            else:
                descriptions.append("with multiple peaks")
        
        # Describe valleys
        valleys_detected = feature_detection.get("valleys_detected", 0)
        valley_depth = feature_detection.get("valley_depth_relative", 0)
        if valleys_detected > 0:
            if valley_depth > 0.3:
                descriptions.append("and deep valleys")
            else:
                descriptions.append("and gentle valleys")
        
        # Describe ridges
        ridge_lines = feature_detection.get("ridge_lines", 0)
        if ridge_lines > 0.2:
            ridge_desc = self.rng.choice(self.feature_descriptors["ridges"])
            descriptions.append(f"featuring {ridge_desc} ridges")
        
        # Describe water features
        water_fraction = feature_detection.get("water_body_fraction", 0)
        if water_fraction > 0.05:
            water_desc = self.rng.choice(self.feature_descriptors["water"])
            descriptions.append(f"with {water_desc} water features")
        
        # Describe caves
        caves_detected = feature_detection.get("caves_detected", 0)
        if caves_detected > 0:
            cave_desc = self.rng.choice(self.feature_descriptors["caves"])
            descriptions.append(f"containing {cave_desc} cave systems")
        
        # Describe erosion
        steep_fraction = slope_analysis.get("steep_area_fraction", 0)
        if steep_fraction > 0.4:
            erosion_desc = self.rng.choice(self.feature_descriptors["erosion"])
            descriptions.append(f"showing {erosion_desc} erosion patterns")
        
        return descriptions
    
    def _describe_complexity(
        self,
        terrain_analysis: Dict[str, Any],
        features: Dict[str, Any]
    ) -> List[str]:
        """Describe terrain complexity and characteristics."""
        
        descriptions = []
        
        classification = terrain_analysis.get("terrain_classification", {})
        characteristics = classification.get("characteristics", [])
        complexity_score = classification.get("complexity_score", 0)
        
        # Add characteristic descriptions
        if "steep" in characteristics:
            descriptions.append("with steep gradients")
        elif "gentle" in characteristics:
            descriptions.append("with gentle gradients")
        
        if "ridged" in characteristics:
            descriptions.append("and prominent ridgelines")
        
        if "water_features" in characteristics:
            descriptions.append("and water channels")
        
        # Add complexity indicators
        if complexity_score > 0.8:
            descriptions.append("displaying complex terrain variations")
        elif complexity_score > 0.6:
            descriptions.append("showing varied topography")
        
        return descriptions
    
    def _join_components(self, components: List[str]) -> str:
        """Join caption components into natural language."""
        
        if not components:
            return "flat terrain"
        
        if len(components) == 1:
            return components[0]
        
        # Join components with appropriate connectors
        result = components[0]
        
        for i, component in enumerate(components[1:], 1):
            if component.startswith("with") or component.startswith("and"):
                result += f" {component}"
            elif component.startswith("featuring") or component.startswith("showing") or component.startswith("containing"):
                result += f" {component}"
            elif component.startswith("displaying"):
                result += f" {component}"
            elif i == len(components) - 1:
                result += f" and {component}"
            elif i == 1:
                result += f" {component}"
            else:
                result += f", {component}"
        
        return result
    
    def generate_batch(
        self,
        analysis_list: List[Dict[str, Any]],
        features_list: List[Dict[str, Any]],
        parameters_list: List[Dict[str, float]]
    ) -> List[str]:
        """Generate captions for a batch of terrain analyses."""
        
        captions = []
        for analysis, features, parameters in zip(analysis_list, features_list, parameters_list):
            caption = self.generate_from_analysis(analysis, features, parameters)
            captions.append(caption)
        
        return captions
    
    def validate_caption_quality(self, caption: str) -> Dict[str, Any]:
        """Validate caption quality and characteristics."""
        
        words = caption.lower().split()
        
        # Check length
        word_count = len(words)
        length_score = 1.0 if 3 <= word_count <= 20 else 0.5
        
        # Check for terrain terms
        terrain_terms = [
            "mountain", "hill", "valley", "peak", "ridge", "plain", "slope",
            "canyon", "ravine", "plateau", "basin", "cliff", "cave", "water"
        ]
        
        terrain_term_count = sum(1 for word in words if any(term in word for term in terrain_terms))
        terrain_score = min(1.0, terrain_term_count / 3.0)
        
        # Check for descriptive adjectives
        descriptive_terms = [
            "steep", "gentle", "deep", "high", "low", "sharp", "smooth",
            "rugged", "dramatic", "rolling", "flat", "curved", "straight"
        ]
        
        descriptive_count = sum(1 for word in words if word in descriptive_terms)
        descriptive_score = min(1.0, descriptive_count / 2.0)
        
        # Overall quality score
        quality_score = (length_score + terrain_score + descriptive_score) / 3.0
        
        return {
            "quality_score": quality_score,
            "word_count": word_count,
            "terrain_terms": terrain_term_count,
            "descriptive_terms": descriptive_count,
            "is_valid": quality_score > 0.6
        }