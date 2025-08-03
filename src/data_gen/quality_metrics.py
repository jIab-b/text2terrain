"""
Quality metrics for dataset generation evaluation.

Provides metrics to assess the quality and diversity of generated training data.
"""

from typing import Dict, List, Any
import numpy as np
from collections import Counter


class QualityMetrics:
    """
    Calculates quality metrics for generated datasets.
    
    Provides comprehensive metrics to evaluate:
    1. Generation success rates
    2. Feature diversity
    3. Terrain type distribution
    4. Validation score distributions
    5. Caption quality metrics
    """
    
    def __init__(self):
        pass
    
    def calculate_dataset_metrics(self, generation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the dataset.
        
        Args:
            generation_stats: Statistics collected during generation
            
        Returns:
            Dictionary of calculated metrics
        """
        
        metrics = {}
        
        # Basic generation metrics
        metrics.update(self._calculate_generation_metrics(generation_stats))
        
        # Feature diversity metrics
        metrics.update(self._calculate_diversity_metrics(generation_stats))
        
        # Quality distribution metrics
        metrics.update(self._calculate_quality_metrics(generation_stats))
        
        # Performance metrics
        metrics.update(self._calculate_performance_metrics(generation_stats))
        
        return metrics
    
    def _calculate_generation_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic generation success metrics."""
        
        total_generated = stats.get("total_generated", 0)
        total_valid = stats.get("total_valid", 0)
        validation_failures = stats.get("validation_failures", 0)
        
        success_rate = total_valid / max(1, total_generated) if total_generated > 0 else 0
        failure_rate = validation_failures / max(1, total_generated) if total_generated > 0 else 0
        
        return {
            "generation_metrics": {
                "total_samples_attempted": total_generated,
                "total_samples_valid": total_valid,
                "validation_failures": validation_failures,
                "success_rate": round(success_rate, 3),
                "failure_rate": round(failure_rate, 3),
                "quality_gate_pass_rate": round(1.0 - failure_rate, 3)
            }
        }
    
    def _calculate_diversity_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate feature and terrain type diversity metrics."""
        
        feature_usage = stats.get("feature_usage", {})
        terrain_types = stats.get("terrain_types", {})
        
        # Feature diversity
        total_feature_usage = sum(feature_usage.values()) if feature_usage else 0
        num_unique_features = len(feature_usage)
        
        feature_distribution = {}
        if total_feature_usage > 0:
            feature_distribution = {
                feature: count / total_feature_usage 
                for feature, count in feature_usage.items()
            }
        
        # Calculate feature entropy (diversity measure)
        feature_entropy = self._calculate_entropy(list(feature_distribution.values()))
        
        # Terrain type diversity
        total_terrain_samples = sum(terrain_types.values()) if terrain_types else 0
        num_terrain_types = len(terrain_types)
        
        terrain_distribution = {}
        if total_terrain_samples > 0:
            terrain_distribution = {
                terrain: count / total_terrain_samples
                for terrain, count in terrain_types.items()
            }
        
        terrain_entropy = self._calculate_entropy(list(terrain_distribution.values()))
        
        return {
            "diversity_metrics": {
                "unique_features_count": num_unique_features,
                "feature_entropy": round(feature_entropy, 3),
                "feature_distribution": feature_distribution,
                "most_common_features": sorted(
                    feature_usage.items(), key=lambda x: x[1], reverse=True
                )[:5],
                "terrain_types_count": num_terrain_types,
                "terrain_entropy": round(terrain_entropy, 3),
                "terrain_distribution": terrain_distribution,
                "terrain_balance_score": self._calculate_balance_score(terrain_distribution)
            }
        }
    
    def _calculate_quality_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality score distribution metrics."""
        
        validation_scores = stats.get("validation_scores", [])
        
        if not validation_scores:
            return {
                "quality_metrics": {
                    "average_validation_score": 0.0,
                    "validation_score_std": 0.0,
                    "high_quality_samples": 0,
                    "medium_quality_samples": 0,
                    "low_quality_samples": 0
                }
            }
        
        validation_array = np.array(validation_scores)
        
        # Score distribution
        high_quality = np.sum(validation_array > 0.8)
        medium_quality = np.sum((validation_array > 0.6) & (validation_array <= 0.8))
        low_quality = np.sum(validation_array <= 0.6)
        
        # Quality percentiles
        percentiles = np.percentile(validation_array, [10, 25, 50, 75, 90])
        
        return {
            "quality_metrics": {
                "average_validation_score": round(float(np.mean(validation_array)), 3),
                "validation_score_std": round(float(np.std(validation_array)), 3),
                "validation_score_min": round(float(np.min(validation_array)), 3),
                "validation_score_max": round(float(np.max(validation_array)), 3),
                "high_quality_samples": int(high_quality),
                "medium_quality_samples": int(medium_quality),
                "low_quality_samples": int(low_quality),
                "quality_percentiles": {
                    "p10": round(float(percentiles[0]), 3),
                    "p25": round(float(percentiles[1]), 3),
                    "p50": round(float(percentiles[2]), 3),
                    "p75": round(float(percentiles[3]), 3),
                    "p90": round(float(percentiles[4]), 3)
                }
            }
        }
    
    def _calculate_performance_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate generation performance metrics."""
        
        generation_times = stats.get("generation_times", [])
        
        if not generation_times:
            return {
                "performance_metrics": {
                    "average_generation_time": 0.0,
                    "generation_time_std": 0.0,
                    "samples_per_second": 0.0
                }
            }
        
        times_array = np.array(generation_times)
        
        avg_time = np.mean(times_array)
        std_time = np.std(times_array)
        samples_per_second = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "performance_metrics": {
                "average_generation_time": round(float(avg_time), 3),
                "generation_time_std": round(float(std_time), 3),
                "min_generation_time": round(float(np.min(times_array)), 3),
                "max_generation_time": round(float(np.max(times_array)), 3),
                "samples_per_second": round(samples_per_second, 2),
                "estimated_time_per_1k_samples": round(avg_time * 1000 / 60, 1)  # minutes
            }
        }
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy for diversity measurement."""
        
        if not probabilities:
            return 0.0
        
        # Filter out zero probabilities to avoid log(0)
        non_zero_probs = [p for p in probabilities if p > 0]
        
        if not non_zero_probs:
            return 0.0
        
        entropy = -sum(p * np.log2(p) for p in non_zero_probs)
        return entropy
    
    def _calculate_balance_score(self, distribution: Dict[str, float]) -> float:
        """Calculate balance score (1.0 = perfectly balanced, 0.0 = completely imbalanced)."""
        
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        
        if len(values) == 1:
            return 1.0  # Single category is perfectly balanced
        
        # Calculate how close to uniform distribution
        uniform_prob = 1.0 / len(values)
        deviations = [abs(p - uniform_prob) for p in values]
        max_possible_deviation = uniform_prob * (len(values) - 1)
        
        if max_possible_deviation == 0:
            return 1.0
        
        balance_score = 1.0 - (sum(deviations) / (2 * max_possible_deviation))
        return round(balance_score, 3)
    
    def generate_quality_report(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable quality report."""
        
        report_lines = []
        report_lines.append("Dataset Quality Report")
        report_lines.append("=" * 50)
        
        # Generation metrics
        gen_metrics = metrics.get("generation_metrics", {})
        report_lines.append(f"\nGeneration Success:")
        report_lines.append(f"  Success Rate: {gen_metrics.get('success_rate', 0):.1%}")
        report_lines.append(f"  Valid Samples: {gen_metrics.get('total_samples_valid', 0)}")
        report_lines.append(f"  Failed Samples: {gen_metrics.get('validation_failures', 0)}")
        
        # Quality metrics
        quality_metrics = metrics.get("quality_metrics", {})
        report_lines.append(f"\nQuality Distribution:")
        report_lines.append(f"  Average Score: {quality_metrics.get('average_validation_score', 0):.3f}")
        report_lines.append(f"  High Quality (>0.8): {quality_metrics.get('high_quality_samples', 0)}")
        report_lines.append(f"  Medium Quality (0.6-0.8): {quality_metrics.get('medium_quality_samples', 0)}")
        report_lines.append(f"  Low Quality (<0.6): {quality_metrics.get('low_quality_samples', 0)}")
        
        # Diversity metrics
        diversity_metrics = metrics.get("diversity_metrics", {})
        report_lines.append(f"\nDataset Diversity:")
        report_lines.append(f"  Unique Features: {diversity_metrics.get('unique_features_count', 0)}")
        report_lines.append(f"  Feature Entropy: {diversity_metrics.get('feature_entropy', 0):.3f}")
        report_lines.append(f"  Terrain Types: {diversity_metrics.get('terrain_types_count', 0)}")
        report_lines.append(f"  Terrain Balance: {diversity_metrics.get('terrain_balance_score', 0):.3f}")
        
        # Performance metrics
        perf_metrics = metrics.get("performance_metrics", {})
        report_lines.append(f"\nGeneration Performance:")
        report_lines.append(f"  Avg Time/Sample: {perf_metrics.get('average_generation_time', 0):.3f}s")
        report_lines.append(f"  Samples/Second: {perf_metrics.get('samples_per_second', 0):.2f}")
        report_lines.append(f"  Est. Time/1k Samples: {perf_metrics.get('estimated_time_per_1k_samples', 0):.1f} min")
        
        # Most common features
        most_common = diversity_metrics.get("most_common_features", [])
        if most_common:
            report_lines.append(f"\nMost Common Features:")
            for feature, count in most_common:
                report_lines.append(f"  {feature}: {count}")
        
        return "\n".join(report_lines)
    
    def assess_dataset_readiness(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if dataset is ready for training."""
        
        gen_metrics = metrics.get("generation_metrics", {})
        quality_metrics = metrics.get("quality_metrics", {})
        diversity_metrics = metrics.get("diversity_metrics", {})
        
        # Readiness criteria
        success_rate = gen_metrics.get("success_rate", 0)
        avg_quality = quality_metrics.get("average_validation_score", 0)
        feature_entropy = diversity_metrics.get("feature_entropy", 0)
        terrain_balance = diversity_metrics.get("terrain_balance_score", 0)
        
        # Assessment thresholds
        readiness_checks = {
            "success_rate": (success_rate > 0.7, f"Success rate: {success_rate:.1%} (need >70%)"),
            "quality_score": (avg_quality > 0.7, f"Quality score: {avg_quality:.3f} (need >0.7)"),
            "feature_diversity": (feature_entropy > 1.0, f"Feature entropy: {feature_entropy:.3f} (need >1.0)"),
            "terrain_balance": (terrain_balance > 0.5, f"Terrain balance: {terrain_balance:.3f} (need >0.5)")
        }
        
        passed_checks = sum(1 for passed, _ in readiness_checks.values() if passed)
        total_checks = len(readiness_checks)
        
        is_ready = passed_checks >= 3  # Need at least 3/4 checks to pass
        
        return {
            "is_ready_for_training": is_ready,
            "readiness_score": passed_checks / total_checks,
            "checks_passed": passed_checks,
            "total_checks": total_checks,
            "detailed_checks": readiness_checks,
            "recommendations": self._generate_readiness_recommendations(readiness_checks)
        }
    
    def _generate_readiness_recommendations(self, checks: Dict[str, tuple]) -> List[str]:
        """Generate recommendations for improving dataset readiness."""
        
        recommendations = []
        
        for check_name, (passed, message) in checks.items():
            if not passed:
                if check_name == "success_rate":
                    recommendations.append("Improve feature validation and terrain generation stability")
                elif check_name == "quality_score":
                    recommendations.append("Tune validation thresholds and caption generation quality")
                elif check_name == "feature_diversity":
                    recommendations.append("Increase variety in feature sampling and terrain templates")
                elif check_name == "terrain_balance":
                    recommendations.append("Balance terrain type distribution in feature templates")
        
        if not recommendations:
            recommendations.append("Dataset meets quality standards and is ready for training!")
        
        return recommendations