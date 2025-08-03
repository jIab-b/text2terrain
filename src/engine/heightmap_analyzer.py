"""
Heightmap analysis for terrain feature detection.

Analyzes generated terrain to detect and validate features
for training data quality and caption generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation


class HeightmapAnalyzer:
    """
    Analyzes heightmaps to detect terrain features.
    
    Provides feature detection and validation for training data generation
    and terrain-to-caption mapping.
    """
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
    
    def analyze(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive terrain analysis.
        
        Args:
            heightmap: Input heightmap to analyze
            
        Returns:
            Dictionary containing detected features and statistics
        """
        
        # Basic elevation statistics
        elevation_stats = self._analyze_elevation(heightmap)
        
        # Slope analysis
        slope_analysis = self._analyze_slopes(heightmap)
        
        # Feature detection
        feature_detection = self._detect_features(heightmap)
        
        # Terrain classification
        terrain_classification = self._classify_terrain(
            elevation_stats, slope_analysis, feature_detection
        )
        
        return {
            "elevation_stats": elevation_stats,
            "slope_analysis": slope_analysis,
            "feature_detection": feature_detection,
            "terrain_classification": terrain_classification,
            "analysis_metadata": {
                "tile_size": self.tile_size,
                "heightmap_shape": heightmap.shape,
                "analysis_version": "1.0"
            }
        }
    
    def _analyze_elevation(self, heightmap: np.ndarray) -> Dict[str, float]:
        """Analyze elevation statistics."""
        
        flat_heightmap = heightmap.flatten()
        
        return {
            "min": float(np.min(flat_heightmap)),
            "max": float(np.max(flat_heightmap)),
            "mean": float(np.mean(flat_heightmap)),
            "median": float(np.median(flat_heightmap)),
            "std": float(np.std(flat_heightmap)),
            "range": float(np.max(flat_heightmap) - np.min(flat_heightmap)),
            "elevation_variance": float(np.var(flat_heightmap))
        }
    
    def _analyze_slopes(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Analyze slope characteristics."""
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(heightmap)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Slope statistics
        flat_slopes = slope_magnitude.flatten()
        
        # Slope direction (aspect)
        slope_direction = np.arctan2(grad_y, grad_x)
        
        # Identify steep areas
        steep_threshold = np.percentile(flat_slopes, 80)
        steep_areas = slope_magnitude > steep_threshold
        steep_fraction = np.sum(steep_areas) / steep_areas.size
        
        return {
            "max_slope": float(np.max(flat_slopes)),
            "mean_slope": float(np.mean(flat_slopes)),
            "median_slope": float(np.median(flat_slopes)),
            "slope_std": float(np.std(flat_slopes)),
            "steep_area_fraction": float(steep_fraction),
            "slope_variance": float(np.var(flat_slopes)),
            "dominant_aspect": float(np.mean(slope_direction)),
            "slope_gradients": {
                "grad_x_mean": float(np.mean(grad_x)),
                "grad_y_mean": float(np.mean(grad_y))
            }
        }
    
    def _detect_features(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect specific terrain features."""
        
        features = {}
        
        # Detect peaks
        features.update(self._detect_peaks(heightmap))
        
        # Detect valleys
        features.update(self._detect_valleys(heightmap))
        
        # Detect ridges
        features.update(self._detect_ridges(heightmap))
        
        # Detect flat areas
        features.update(self._detect_flat_areas(heightmap))
        
        # Detect water bodies (low areas)
        features.update(self._detect_water_bodies(heightmap))
        
        # Detect caves (represented as local minima in elevated areas)
        features.update(self._detect_caves(heightmap))
        
        return features
    
    def _detect_peaks(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect mountain peaks and elevated points."""
        
        # Use local maxima detection
        from scipy.ndimage import maximum_filter
        
        # Local maxima within neighborhood
        neighborhood_size = max(5, self.tile_size // 50)
        local_maxima = maximum_filter(heightmap, size=neighborhood_size) == heightmap
        
        # Filter by height threshold
        height_threshold = np.percentile(heightmap, 85)
        significant_peaks = local_maxima & (heightmap > height_threshold)
        
        # Count and analyze peaks
        peak_count = np.sum(significant_peaks)
        
        if peak_count > 0:
            peak_heights = heightmap[significant_peaks]
            peak_locations = np.where(significant_peaks)
            
            return {
                "peaks_detected": int(peak_count),
                "peak_height_mean": float(np.mean(peak_heights)),
                "peak_height_max": float(np.max(peak_heights)),
                "peak_locations": [(int(y), int(x)) for y, x in zip(peak_locations[0], peak_locations[1])],
                "peak_prominence": float(np.mean(peak_heights) - np.mean(heightmap))
            }
        else:
            return {
                "peaks_detected": 0,
                "peak_height_mean": 0.0,
                "peak_height_max": 0.0,
                "peak_locations": [],
                "peak_prominence": 0.0
            }
    
    def _detect_valleys(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect valleys and low-lying channels."""
        
        # Use local minima detection
        from scipy.ndimage import minimum_filter
        
        # Local minima within neighborhood
        neighborhood_size = max(5, self.tile_size // 50)
        local_minima = minimum_filter(heightmap, size=neighborhood_size) == heightmap
        
        # Filter by depth threshold
        depth_threshold = np.percentile(heightmap, 15)
        significant_valleys = local_minima & (heightmap < depth_threshold)
        
        # Count and analyze valleys
        valley_count = np.sum(significant_valleys)
        
        if valley_count > 0:
            valley_depths = heightmap[significant_valleys]
            valley_locations = np.where(significant_valleys)
            
            # Calculate valley depth relative to surroundings
            valley_depth_relative = np.mean(heightmap) - np.mean(valley_depths)
            
            return {
                "valleys_detected": int(valley_count),
                "valley_depth_mean": float(np.mean(valley_depths)),
                "valley_depth_min": float(np.min(valley_depths)),
                "valley_locations": [(int(y), int(x)) for y, x in zip(valley_locations[0], valley_locations[1])],
                "valley_depth_relative": float(valley_depth_relative)
            }
        else:
            return {
                "valleys_detected": 0,
                "valley_depth_mean": 0.0,
                "valley_depth_min": 0.0,
                "valley_locations": [],
                "valley_depth_relative": 0.0
            }
    
    def _detect_ridges(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect ridge lines and elevated linear features."""
        
        # Calculate second derivatives (curvature)
        grad_y, grad_x = np.gradient(heightmap)
        grad_yy, grad_yx = np.gradient(grad_y)
        grad_xy, grad_xx = np.gradient(grad_x)
        
        # Ridge detection using eigenvalues of Hessian matrix
        # Ridge points have one large negative eigenvalue
        trace = grad_xx + grad_yy
        determinant = grad_xx * grad_yy - grad_xy * grad_yx
        
        # Eigenvalues of 2x2 Hessian
        lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4 * determinant))
        lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4 * determinant))
        
        # Ridge condition: one eigenvalue strongly negative, other near zero
        ridge_strength = np.abs(np.minimum(lambda1, lambda2))
        
        # Threshold for ridge detection
        ridge_threshold = np.percentile(ridge_strength, 90)
        ridge_areas = ridge_strength > ridge_threshold
        
        # Also require elevated areas
        elevation_threshold = np.percentile(heightmap, 60)
        ridges = ridge_areas & (heightmap > elevation_threshold)
        
        ridge_fraction = np.sum(ridges) / ridges.size
        
        return {
            "ridge_lines": float(ridge_fraction),
            "ridge_strength_mean": float(np.mean(ridge_strength[ridges])) if np.any(ridges) else 0.0,
            "ridge_area_fraction": float(ridge_fraction)
        }
    
    def _detect_flat_areas(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect flat areas in the terrain."""
        
        # Calculate slope magnitude
        grad_y, grad_x = np.gradient(heightmap)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Flat areas have low slope
        flat_threshold = np.percentile(slope_magnitude, 20)
        flat_areas = slope_magnitude < flat_threshold
        
        flat_fraction = np.sum(flat_areas) / flat_areas.size
        
        # Find connected flat regions
        labeled_flat, num_regions = label(flat_areas)
        
        region_sizes = []
        largest_flat_region = 0
        
        if num_regions > 0:
            # Analyze size of flat regions
            region_sizes = [np.sum(labeled_flat == i) for i in range(1, num_regions + 1)]
            largest_flat_region = max(region_sizes) if region_sizes else 0
        
        return {
            "flat_area_fraction": float(flat_fraction),
            "flat_regions_count": int(num_regions),
            "largest_flat_region": int(largest_flat_region),
            "average_flat_region_size": float(np.mean(region_sizes)) if region_sizes else 0.0
        }
    
    def _detect_water_bodies(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect potential water bodies (very low, flat areas)."""
        
        # Water bodies are low and relatively flat
        low_threshold = np.percentile(heightmap, 10)
        low_areas = heightmap < low_threshold
        
        # Calculate local slope in low areas
        grad_y, grad_x = np.gradient(heightmap)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Water areas should be flat
        flat_threshold = np.percentile(slope_magnitude, 15)
        flat_areas = slope_magnitude < flat_threshold
        
        # Water bodies are low AND flat
        water_areas = low_areas & flat_areas
        
        # Find connected water bodies
        labeled_water, num_water_bodies = label(water_areas)
        
        water_fraction = np.sum(water_areas) / water_areas.size
        
        water_sizes = []
        largest_water_body = 0
        
        if num_water_bodies > 0:
            water_sizes = [np.sum(labeled_water == i) for i in range(1, num_water_bodies + 1)]
            largest_water_body = max(water_sizes)
        
        return {
            "water_body_fraction": float(water_fraction),
            "water_bodies_count": int(num_water_bodies),
            "largest_water_body": int(largest_water_body),
            "average_water_depth": float(np.mean(heightmap[water_areas])) if np.any(water_areas) else 0.0
        }
    
    def _detect_caves(self, heightmap: np.ndarray) -> Dict[str, Any]:
        """Detect cave-like features (local depressions in elevated areas)."""
        
        # Caves are local minima in otherwise elevated terrain
        from scipy.ndimage import minimum_filter
        
        # Find elevated areas first
        elevation_threshold = np.percentile(heightmap, 50)
        elevated_areas = heightmap > elevation_threshold
        
        # Find local minima within elevated areas
        neighborhood_size = max(3, self.tile_size // 80)
        local_minima = minimum_filter(heightmap, size=neighborhood_size) == heightmap
        
        # Caves are local minima in elevated areas that are significantly lower than surroundings
        potential_caves = local_minima & elevated_areas
        
        cave_count = 0
        cave_depths = []
        
        if np.any(potential_caves):
            cave_locations = np.where(potential_caves)
            
            for y, x in zip(cave_locations[0], cave_locations[1]):
                # Check if this point is significantly lower than its neighborhood
                y_min = max(0, y - neighborhood_size//2)
                y_max = min(heightmap.shape[0], y + neighborhood_size//2 + 1)
                x_min = max(0, x - neighborhood_size//2)
                x_max = min(heightmap.shape[1], x + neighborhood_size//2 + 1)
                
                neighborhood = heightmap[y_min:y_max, x_min:x_max]
                center_height = heightmap[y, x]
                
                # Cave depth relative to neighborhood
                depth_below_neighborhood = np.mean(neighborhood) - center_height
                
                if depth_below_neighborhood > np.std(heightmap) * 0.5:  # Significant depression
                    cave_count += 1
                    cave_depths.append(depth_below_neighborhood)
        
        return {
            "caves_detected": int(cave_count),
            "cave_depth_mean": float(np.mean(cave_depths)) if cave_depths else 0.0,
            "cave_depth_max": float(np.max(cave_depths)) if cave_depths else 0.0,
            "cave_area_fraction": float(cave_count / (heightmap.size / 1000))  # Normalized by area
        }
    
    def _classify_terrain(
        self,
        elevation_stats: Dict,
        slope_analysis: Dict,
        feature_detection: Dict
    ) -> Dict[str, Any]:
        """Classify overall terrain type based on analysis."""
        
        # Primary terrain classification
        terrain_type = "plains"  # Default
        confidence = 0.5
        
        # Mountain classification
        if (feature_detection["peaks_detected"] > 2 and 
            slope_analysis["steep_area_fraction"] > 0.3 and
            elevation_stats["range"] > elevation_stats["std"] * 2):
            terrain_type = "mountains"
            confidence = 0.8
        
        # Valley classification  
        elif (feature_detection["valleys_detected"] > 1 and
              slope_analysis["mean_slope"] > elevation_stats["std"] * 0.5):
            terrain_type = "valleys"
            confidence = 0.7
        
        # Hills classification
        elif (elevation_stats["range"] > elevation_stats["std"] and
              slope_analysis["mean_slope"] > elevation_stats["std"] * 0.3):
            terrain_type = "hills"
            confidence = 0.6
        
        # Water/wetland classification
        elif feature_detection["water_body_fraction"] > 0.1:
            terrain_type = "wetlands"
            confidence = 0.7
        
        # Flat plains
        elif (feature_detection["flat_area_fraction"] > 0.6 and
              slope_analysis["mean_slope"] < elevation_stats["std"] * 0.2):
            terrain_type = "plains"
            confidence = 0.8
        
        # Secondary characteristics
        characteristics = []
        
        if feature_detection["ridge_lines"] > 0.1:
            characteristics.append("ridged")
        
        if feature_detection["caves_detected"] > 0:
            characteristics.append("caves")
        
        if feature_detection["water_bodies_count"] > 0:
            characteristics.append("water_features")
        
        if slope_analysis["steep_area_fraction"] > 0.4:
            characteristics.append("steep")
        elif slope_analysis["steep_area_fraction"] < 0.1:
            characteristics.append("gentle")
        
        return {
            "primary_type": terrain_type,
            "confidence": float(confidence),
            "characteristics": characteristics,
            "complexity_score": float(
                elevation_stats["elevation_variance"] + 
                slope_analysis["slope_variance"] + 
                feature_detection["peaks_detected"] * 0.1 +
                feature_detection["valleys_detected"] * 0.1
            )
        }