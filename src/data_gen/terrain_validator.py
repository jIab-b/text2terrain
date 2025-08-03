"""
Terrain validation for training data quality assurance.

Validates that generated terrain matches caption descriptions
and feature configurations for high-quality training data.
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np


class TerrainValidator:
    """
    Validates consistency between terrain, captions, and features.
    
    Ensures training data quality by verifying that:
    1. Generated terrain matches feature specifications
    2. Captions accurately describe terrain characteristics
    3. Feature parameters produce expected results
    """
    
    def __init__(self):
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Initialize validation rules and thresholds."""
        
        # Feature validation thresholds
        self.feature_thresholds = {
            "mountain_peaks": {
                "min_peaks": 1,
                "min_elevation_range": 0.3,
                "min_steep_fraction": 0.2
            },
            "steep_peaks": {
                "min_peaks": 1,
                "min_elevation_range": 0.5,
                "min_steep_fraction": 0.4
            },
            "deep_valleys": {
                "min_valleys": 1,
                "min_valley_depth": 0.2,
                "max_flat_fraction": 0.8
            },
            "cave_systems": {
                "min_caves": 1,
                "min_cave_depth": 0.1
            },
            "water_features": {
                "min_water_fraction": 0.02,
                "min_water_bodies": 1
            },
            "rolling_hills": {
                "max_peaks": 5,
                "min_elevation_range": 0.1,
                "max_steep_fraction": 0.3
            }
        }
        
        # Caption keyword mapping
        self.caption_keywords = {
            "mountain": ["peaks_detected", "elevation_range", "steep_area_fraction"],
            "peak": ["peaks_detected", "peak_prominence"],
            "steep": ["steep_area_fraction", "max_slope"],
            "valley": ["valleys_detected", "valley_depth_relative"],
            "deep": ["valley_depth_relative", "elevation_range"],
            "gentle": ["steep_area_fraction", "mean_slope"],
            "rolling": ["elevation_variance", "flat_area_fraction"],
            "flat": ["flat_area_fraction", "mean_slope"],
            "cave": ["caves_detected", "cave_depth_mean"],
            "water": ["water_body_fraction", "water_bodies_count"],
            "ridge": ["ridge_lines", "ridge_strength_mean"],
            "rugged": ["slope_variance", "elevation_variance"],
            "smooth": ["slope_variance", "flat_area_fraction"]
        }
    
    def validate_consistency(
        self,
        heightmap: np.ndarray,
        caption: str,
        features: Dict[str, Any],
        terrain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate consistency between terrain, caption, and features.
        
        Args:
            heightmap: Generated heightmap
            caption: Generated caption
            features: Feature configuration
            terrain_analysis: Terrain analysis results
            
        Returns:
            Validation result with score and details
        """
        
        validation_results = []
        
        # 1. Validate features match terrain
        feature_validation = self._validate_features(features, terrain_analysis)
        validation_results.append(("features", feature_validation))
        
        # 2. Validate caption matches terrain
        caption_validation = self._validate_caption(caption, terrain_analysis)
        validation_results.append(("caption", caption_validation))
        
        # 3. Validate terrain quality
        quality_validation = self._validate_terrain_quality(heightmap, terrain_analysis)
        validation_results.append(("quality", quality_validation))
        
        # 4. Cross-validate caption and features
        cross_validation = self._cross_validate_caption_features(caption, features)
        validation_results.append(("cross", cross_validation))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(validation_results)
        
        # Determine if sample is valid
        is_valid = overall_score > 0.6 and all(result[1]["score"] > 0.4 for result in validation_results)
        
        return {
            "is_valid": is_valid,
            "score": overall_score,
            "feature_score": feature_validation["score"],
            "caption_score": caption_validation["score"],
            "quality_score": quality_validation["score"],
            "cross_score": cross_validation["score"],
            "details": {result[0]: result[1] for result in validation_results},
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def _validate_features(
        self,
        features: Dict[str, Any],
        terrain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that terrain matches specified features."""
        
        feature_detection = terrain_analysis.get("feature_detection", {})
        elevation_stats = terrain_analysis.get("elevation_stats", {})
        slope_analysis = terrain_analysis.get("slope_analysis", {})
        
        validations = []
        primary_features = features.get("primary_features", [])
        
        for feature in primary_features:
            if feature in self.feature_thresholds:
                thresholds = self.feature_thresholds[feature]
                feature_score = self._validate_single_feature(
                    feature, thresholds, feature_detection, elevation_stats, slope_analysis
                )
                validations.append((feature, feature_score))
        
        if not validations:
            return {"score": 0.5, "details": "No features to validate"}
        
        # Average feature validation scores
        avg_score = np.mean([score for _, score in validations])
        
        return {
            "score": avg_score,
            "details": validations,
            "feature_count": len(validations)
        }
    
    def _validate_single_feature(
        self,
        feature: str,
        thresholds: Dict[str, float],
        feature_detection: Dict[str, Any],
        elevation_stats: Dict[str, float],
        slope_analysis: Dict[str, float]
    ) -> float:
        """Validate a single feature against thresholds."""
        
        score = 1.0
        
        # Check each threshold
        for threshold_name, threshold_value in thresholds.items():
            actual_value = self._get_validation_value(
                threshold_name, feature_detection, elevation_stats, slope_analysis
            )
            
            if threshold_name.startswith("min_"):
                if actual_value < threshold_value:
                    score *= 0.5  # Penalty for not meeting minimum
            elif threshold_name.startswith("max_"):
                if actual_value > threshold_value:
                    score *= 0.5  # Penalty for exceeding maximum
        
        return score
    
    def _get_validation_value(
        self,
        metric_name: str,
        feature_detection: Dict[str, Any],
        elevation_stats: Dict[str, float],
        slope_analysis: Dict[str, float]
    ) -> float:
        """Get validation metric value from terrain analysis."""
        
        # Map metric names to analysis results
        value_mapping = {
            "peaks": feature_detection.get("peaks_detected", 0),
            "elevation_range": elevation_stats.get("range", 0),
            "steep_fraction": slope_analysis.get("steep_area_fraction", 0),
            "valleys": feature_detection.get("valleys_detected", 0),
            "valley_depth": feature_detection.get("valley_depth_relative", 0),
            "caves": feature_detection.get("caves_detected", 0),
            "cave_depth": feature_detection.get("cave_depth_mean", 0),
            "water_fraction": feature_detection.get("water_body_fraction", 0),
            "water_bodies": feature_detection.get("water_bodies_count", 0),
            "flat_fraction": feature_detection.get("flat_area_fraction", 0)
        }
        
        # Extract the key part of the metric name
        for key, value in value_mapping.items():
            if key in metric_name:
                return value
        
        return 0.0
    
    def _validate_caption(
        self,
        caption: str,
        terrain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that caption accurately describes terrain."""
        
        caption_lower = caption.lower()
        feature_detection = terrain_analysis.get("feature_detection", {})
        elevation_stats = terrain_analysis.get("elevation_stats", {})
        slope_analysis = terrain_analysis.get("slope_analysis", {})
        
        validations = []
        
        # Check each keyword in caption
        for keyword, metrics in self.caption_keywords.items():
            if keyword in caption_lower:
                keyword_score = self._validate_caption_keyword(
                    keyword, metrics, feature_detection, elevation_stats, slope_analysis
                )
                validations.append((keyword, keyword_score))
        
        if not validations:
            return {"score": 0.3, "details": "No terrain keywords found in caption"}
        
        # Average keyword validation scores
        avg_score = np.mean([score for _, score in validations])
        
        # Bonus for descriptive captions
        word_count = len(caption.split())
        descriptive_bonus = min(0.2, (word_count - 3) * 0.02) if word_count > 3 else 0
        
        final_score = min(1.0, avg_score + descriptive_bonus)
        
        return {
            "score": final_score,
            "details": validations,
            "keyword_count": len(validations),
            "word_count": word_count
        }
    
    def _validate_caption_keyword(
        self,
        keyword: str,
        metrics: List[str],
        feature_detection: Dict[str, Any],
        elevation_stats: Dict[str, float],
        slope_analysis: Dict[str, float]
    ) -> float:
        """Validate a single caption keyword against terrain metrics."""
        
        # Expected thresholds for keywords
        keyword_thresholds = {
            "mountain": {"peaks_detected": 1, "elevation_range": 0.3, "steep_area_fraction": 0.2},
            "peak": {"peaks_detected": 1, "peak_prominence": 0.1},
            "steep": {"steep_area_fraction": 0.3, "max_slope": 0.5},
            "valley": {"valleys_detected": 1, "valley_depth_relative": 0.1},
            "deep": {"valley_depth_relative": 0.3, "elevation_range": 0.4},
            "gentle": {"steep_area_fraction": 0.2, "mean_slope": 0.3},
            "flat": {"flat_area_fraction": 0.4, "mean_slope": 0.1},
            "cave": {"caves_detected": 1, "cave_depth_mean": 0.1},
            "water": {"water_body_fraction": 0.02, "water_bodies_count": 1},
            "ridge": {"ridge_lines": 0.1, "ridge_strength_mean": 0.1}
        }
        
        if keyword not in keyword_thresholds:
            return 0.5  # Neutral score for unknown keywords
        
        thresholds = keyword_thresholds[keyword]
        score = 1.0
        
        for metric, threshold in thresholds.items():
            actual_value = self._get_metric_value(
                metric, feature_detection, elevation_stats, slope_analysis
            )
            
            if actual_value < threshold:
                score *= 0.7  # Penalty for keyword not matching terrain
        
        return score
    
    def _get_metric_value(
        self,
        metric: str,
        feature_detection: Dict[str, Any],
        elevation_stats: Dict[str, float],
        slope_analysis: Dict[str, float]
    ) -> float:
        """Get metric value from terrain analysis."""
        
        # Try feature detection first
        if metric in feature_detection:
            return feature_detection[metric]
        
        # Try elevation stats
        if metric in elevation_stats:
            return elevation_stats[metric]
        
        # Try slope analysis
        if metric in slope_analysis:
            return slope_analysis[metric]
        
        return 0.0
    
    def _validate_terrain_quality(
        self,
        heightmap: np.ndarray,
        terrain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate overall terrain quality."""
        
        quality_score = 1.0
        issues = []
        
        # Check for reasonable elevation range
        elevation_stats = terrain_analysis.get("elevation_stats", {})
        elevation_range = elevation_stats.get("range", 0)
        
        if elevation_range < 0.01:
            quality_score *= 0.5
            issues.append("Very low elevation variation")
        elif elevation_range > 10000:
            quality_score *= 0.7
            issues.append("Extreme elevation variation")
        
        # Check for reasonable slope distribution
        slope_analysis = terrain_analysis.get("slope_analysis", {})
        max_slope = slope_analysis.get("max_slope", 0)
        
        if max_slope > 5.0:
            quality_score *= 0.8
            issues.append("Extremely steep slopes")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(heightmap)) or np.any(np.isinf(heightmap)):
            quality_score = 0.0
            issues.append("Invalid heightmap values (NaN/Inf)")
        
        # Check for reasonable terrain complexity
        terrain_classification = terrain_analysis.get("terrain_classification", {})
        complexity = terrain_classification.get("complexity_score", 0.5)
        
        if complexity < 0.1:
            quality_score *= 0.8
            issues.append("Very low terrain complexity")
        
        return {
            "score": quality_score,
            "issues": issues,
            "elevation_range": elevation_range,
            "max_slope": max_slope,
            "complexity": complexity
        }
    
    def _cross_validate_caption_features(
        self,
        caption: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cross-validate caption keywords against specified features."""
        
        caption_lower = caption.lower()
        primary_features = features.get("primary_features", [])
        
        # Extract terrain type indicators from caption
        caption_indicators = []
        
        if any(word in caption_lower for word in ["mountain", "peak", "steep"]):
            caption_indicators.append("mountain")
        if any(word in caption_lower for word in ["valley", "gorge", "canyon"]):
            caption_indicators.append("valley")
        if any(word in caption_lower for word in ["hill", "rolling", "gentle"]):
            caption_indicators.append("hill")
        if any(word in caption_lower for word in ["flat", "plain", "level"]):
            caption_indicators.append("plain")
        if any(word in caption_lower for word in ["cave", "underground", "hollow"]):
            caption_indicators.append("cave")
        if any(word in caption_lower for word in ["water", "river", "lake"]):
            caption_indicators.append("water")
        
        # Check consistency with features
        consistency_score = 1.0
        
        # Feature to indicator mapping
        feature_indicators = {
            "mountain_peaks": "mountain",
            "steep_peaks": "mountain", 
            "rolling_hills": "hill",
            "deep_valleys": "valley",
            "river_valleys": "valley",
            "cave_systems": "cave",
            "water_features": "water"
        }
        
        expected_indicators = []
        for feature in primary_features:
            if feature in feature_indicators:
                expected_indicators.append(feature_indicators[feature])
        
        # Check if caption indicators match feature indicators
        matches = len(set(caption_indicators) & set(expected_indicators))
        mismatches = len(set(caption_indicators) - set(expected_indicators))
        
        if mismatches > 0:
            consistency_score *= 0.7
        
        if matches == 0 and expected_indicators:
            consistency_score *= 0.5
        
        return {
            "score": consistency_score,
            "caption_indicators": caption_indicators,
            "expected_indicators": expected_indicators,
            "matches": matches,
            "mismatches": mismatches
        }
    
    def _calculate_overall_score(self, validation_results: List[Tuple[str, Dict]]) -> float:
        """Calculate weighted overall validation score."""
        
        # Weights for different validation aspects
        weights = {
            "features": 0.3,
            "caption": 0.3,
            "quality": 0.2,
            "cross": 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for validation_type, result in validation_results:
            if validation_type in weights:
                weight = weights[validation_type]
                score = result.get("score", 0.0)
                weighted_score += weight * score
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, validation_results: List[Tuple[str, Dict]]) -> List[str]:
        """Generate recommendations for improving sample quality."""
        
        recommendations = []
        
        for validation_type, result in validation_results:
            score = result.get("score", 0.0)
            
            if score < 0.5:
                if validation_type == "features":
                    recommendations.append("Adjust feature parameters to better match terrain output")
                elif validation_type == "caption":
                    recommendations.append("Improve caption accuracy to better describe terrain")
                elif validation_type == "quality":
                    recommendations.append("Check terrain generation parameters for quality issues")
                elif validation_type == "cross":
                    recommendations.append("Ensure caption keywords match specified features")
        
        return recommendations