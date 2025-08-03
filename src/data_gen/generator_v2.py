"""
Enhanced dataset generator with terrain validation and feature-based approach.

Generates training data with actual terrain generation and validation
to ensure LLM learns accurate terrain-to-text mappings.
"""

import json
import argparse
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm

from ..engine import TerrainComposer, HeightmapAnalyzer
from ..compatibility import LegacyAdapter, JSONValidator
from .feature_captions import FeatureCaptionGenerator
from .terrain_validator import TerrainValidator
from .quality_metrics import QualityMetrics


class DatasetGeneratorV2:
    """
    Enhanced dataset generator with terrain validation.
    
    Generates training data by:
    1. Sampling feature-based terrain configurations
    2. Generating actual heightmaps
    3. Analyzing terrain features
    4. Generating captions from analysis
    5. Validating terrain-caption consistency
    6. Converting to legacy format for compatibility
    """
    
    def __init__(
        self,
        output_dir: str,
        tile_size: int = 256,
        seed: int = 42,
        mode: str = "feature_based"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_size = tile_size
        self.mode = mode
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Initialize components
        self.terrain_composer = TerrainComposer(tile_size=tile_size)
        self.heightmap_analyzer = HeightmapAnalyzer(tile_size=tile_size)
        self.caption_generator = FeatureCaptionGenerator(seed=seed)
        self.terrain_validator = TerrainValidator()
        self.quality_metrics = QualityMetrics()
        self.legacy_adapter = LegacyAdapter()
        self.json_validator = JSONValidator()
        
        # Track generation statistics
        self.stats = {
            "total_generated": 0,
            "total_valid": 0,
            "validation_failures": 0,
            "feature_usage": {},
            "terrain_types": {},
            "generation_times": [],
            "validation_scores": []
        }
        
        # Feature configurations for sampling
        self._setup_feature_configurations()
    
    def _setup_feature_configurations(self):
        """Initialize feature configuration templates for sampling."""
        
        self.feature_templates = {
            "mountain_terrain": {
                "terrain_type": "mountains",
                "primary_features": ["mountain_peaks", "steep_peaks"],
                "secondary_features": ["ridges", "erosion"],
                "biome": "alpine",
                "complexity": 0.8,
                "parameters": {
                    "mountain_height": (0.6, 1.0),
                    "mountain_steepness": (0.6, 0.9),
                    "peak_count": (2, 5),
                    "ridge_prominence": (0.5, 0.8)
                }
            },
            "valley_terrain": {
                "terrain_type": "valleys",
                "primary_features": ["deep_valleys", "river_valleys"],
                "secondary_features": ["erosion", "water_features"],
                "biome": "temperate",
                "complexity": 0.6,
                "parameters": {
                    "valley_depth": (0.4, 0.8),
                    "valley_width": (0.3, 0.6),
                    "valley_count": (1, 3),
                    "river_depth": (0.2, 0.4)
                }
            },
            "hills_terrain": {
                "terrain_type": "hills",
                "primary_features": ["rolling_hills"],
                "secondary_features": ["gentle_slopes"],
                "biome": "grassland",
                "complexity": 0.4,
                "parameters": {
                    "mountain_height": (0.3, 0.6),
                    "mountain_steepness": (0.2, 0.5),
                    "base_frequency": (0.005, 0.02)
                }
            },
            "cave_terrain": {
                "terrain_type": "hills",
                "primary_features": ["cave_systems"],
                "secondary_features": ["underground"],
                "biome": "temperate",
                "complexity": 0.7,
                "parameters": {
                    "cave_density": (0.3, 0.7),
                    "cave_size": (0.4, 0.8),
                    "tunnel_width": (0.2, 0.5),
                    "cave_depth": (0.3, 0.6)
                }
            },
            "water_terrain": {
                "terrain_type": "plains",
                "primary_features": ["water_features"],
                "secondary_features": ["erosion"],
                "biome": "wetlands",
                "complexity": 0.5,
                "parameters": {
                    "river_depth": (0.3, 0.6),
                    "river_width": (0.2, 0.5),
                    "river_count": (1, 3),
                    "lake_count": (0, 2)
                }
            },
            "desert_terrain": {
                "terrain_type": "plains",
                "primary_features": ["sand_dunes"],
                "secondary_features": ["rocky_outcrops"],
                "biome": "desert",
                "complexity": 0.4,
                "parameters": {
                    "biome_type": "desert",
                    "biome_strength": (0.6, 0.9),
                    "base_frequency": (0.01, 0.03)
                }
            },
            "mixed_terrain": {
                "terrain_type": "mixed",
                "primary_features": ["mountain_peaks", "deep_valleys"],
                "secondary_features": ["ridges", "erosion", "water_features"],
                "biome": "temperate",
                "complexity": 0.9,
                "parameters": {
                    "mountain_height": (0.5, 0.8),
                    "valley_depth": (0.4, 0.7),
                    "ridge_prominence": (0.4, 0.7)
                }
            }
        }
    
    def sample_terrain_config(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Sample a terrain configuration for generation.
        
        Returns:
            Tuple of (features, parameters)
        """
        
        # Choose random terrain template
        template_name = self.rng.choice(list(self.feature_templates.keys()))
        template = self.feature_templates[template_name].copy()
        
        # Sample parameters within ranges
        parameters = {}
        param_ranges = template.get("parameters", {})
        
        for param_name, value_range in param_ranges.items():
            if isinstance(value_range, tuple) and len(value_range) == 2:
                min_val, max_val = value_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    parameters[param_name] = self.rng.randint(min_val, max_val)
                else:
                    parameters[param_name] = self.rng.uniform(min_val, max_val)
            else:
                parameters[param_name] = value_range
        
        # Add base terrain parameters
        base_params = {
            "base_frequency": self.rng.uniform(0.005, 0.02),
            "detail_octaves": self.rng.randint(3, 6),
            "detail_persistence": self.rng.uniform(0.4, 0.7),
            "height_scale": self.rng.uniform(800, 2500)
        }
        parameters.update(base_params)
        
        # Extract features (remove parameters from template)
        features = {k: v for k, v in template.items() if k != "parameters"}
        
        # Add some randomness to features
        if self.rng.random() < 0.3:
            # Sometimes add extra secondary features
            extra_features = ["warping", "detail_noise", "smoothing"]
            extra = self.rng.choice(extra_features)
            if extra not in features.get("secondary_features", []):
                features.setdefault("secondary_features", []).append(extra)
        
        return features, parameters
    
    def sample_world_coordinates(self) -> Tuple[int, int]:
        """Sample world coordinates for terrain generation."""
        
        # Generate coordinates across different regions
        world_scale = 20  # Creates coordinates from -5,120 to +5,120
        world_x = self.rng.randint(-world_scale, world_scale) * self.tile_size
        world_y = self.rng.randint(-world_scale, world_scale) * self.tile_size
        
        return world_x, world_y
    
    def generate_seeds(self, sample_id: int, num_features: int) -> List[int]:
        """Generate deterministic seeds for reproducibility."""
        
        seeds = []
        for i in range(num_features):
            seed = ((sample_id * 10007 + i * 1013) % (2**31 - 1))
            seeds.append(seed)
        
        return seeds
    
    def generate_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Generate a single training sample with validation.
        
        Args:
            sample_id: Unique identifier for this sample
            
        Returns:
            Complete sample dictionary or None if validation fails
        """
        
        start_time = time.time()
        
        calls = []
        try:
            # 1. Sample feature-based configuration
            features, parameters = self.sample_terrain_config()
            calls.append({"id":"call_0","type":"function","function":{"name":"sample_terrain_config","arguments":json.dumps({"features":features,"parameters":parameters})}})
            
            # 2. Sample world coordinates
            world_x, world_y = self.sample_world_coordinates()
            calls.append({"id":"call_1","type":"function","function":{"name":"sample_world_coordinates","arguments":json.dumps({"world_x":world_x,"world_y":world_y})}})
            
            # 3. Generate seeds
            num_features = len(features.get("primary_features", [])) + len(features.get("secondary_features", []))
            seeds = self.generate_seeds(sample_id, max(1, num_features))
            calls.append({"id":"call_2","type":"function","function":{"name":"generate_seeds","arguments":json.dumps({"sample_id":sample_id,"num_features":num_features,"seeds":seeds})}})
            global_seed = (sample_id * 48271) % (2**31 - 1)
            
            # 4. Generate ACTUAL heightmap
            heightmap = self.terrain_composer.generate_heightmap(
                world_x=world_x,
                world_y=world_y,
                features=features,
                parameters=parameters,
                seeds=seeds,
                global_seed=global_seed,
                legacy_mode=False
            )
            calls.append({"id":"call_3","type":"function","function":{"name":"generate_heightmap","arguments":json.dumps({"world_x":world_x,"world_y":world_y,"parameters":parameters,"seeds":seeds,"global_seed":global_seed})}})
            
            # 5. Analyze generated terrain
            terrain_analysis = self.heightmap_analyzer.analyze(heightmap)
            calls.append({"id":"call_4","type":"function","function":{"name":"analyze_heightmap","arguments":json.dumps({"terrain_analysis":terrain_analysis})}})
            
            # 6. Generate caption from ACTUAL terrain features
            caption = self.caption_generator.generate_from_analysis(
                terrain_analysis, features, parameters
            )
            calls.append({"id":"call_5","type":"function","function":{"name":"generate_caption","arguments":json.dumps({"terrain_analysis":terrain_analysis,"features":features,"parameters":parameters})}})
            
            # 7. Validate terrain matches caption
            validation_result = self.terrain_validator.validate_consistency(
                heightmap, caption, features, terrain_analysis
            )
            calls.append({"id":"call_6","type":"function","function":{"name":"validate_consistency","arguments":json.dumps({"caption":caption,"features":features,"terrain_analysis":terrain_analysis})},"result":validation_result})
            
            if not validation_result["is_valid"] or validation_result["score"] < 0.6:
                self.stats["validation_failures"] += 1
                return None  # Skip samples that don't validate
            
            # 8. Convert to legacy format for compatibility
            legacy_data = self.legacy_adapter.convert_to_legacy_format(
                features, parameters, world_x, world_y, seeds, global_seed, self.tile_size
            )
            calls.append({"id":"call_7","type":"function","function":{"name":"convert_to_legacy_format","arguments":json.dumps({"features":features,"parameters":parameters,"world_x":world_x,"world_y":world_y,"seeds":seeds,"global_seed":global_seed,"tile_size":self.tile_size})},"result":legacy_data})
            
            # 9. Create grid continuity info
            grid_continuity = {
                "blend_edges": True,
                "overlap_size": 16,
                "tile_coordinates": (world_x // self.tile_size, world_y // self.tile_size)
            }
            calls.append({"id":"call_8","type":"function","function":{"name":"compute_grid_continuity","arguments":json.dumps(grid_continuity)}})
            
            # 10. Create enhanced JSON with backward compatibility
            enhanced_json = self.legacy_adapter.create_enhanced_json(
                legacy_data, features, terrain_analysis, grid_continuity
            )
            calls.append({"id":"call_3","type":"function","function":{"name":"generate_heightmap","arguments":json.dumps(enhanced_json)}})
            
            # 11. Validate JSON format
            is_valid, validation_errors = self.json_validator.validate_enhanced_format(enhanced_json)
            calls.append({"id":"call_10","type":"function","function":{"name":"validate_enhanced_format","arguments":json.dumps({"is_valid":is_valid,"errors":validation_errors})}})
            if not is_valid:
                print(f"JSON validation failed for sample {sample_id}: {validation_errors}")
                return None
            
            # 12. Create final sample structure
            generation_time = time.time() - start_time
            heightmap_hash = self._hash_heightmap(heightmap)
            
            sample = {
                "messages": [
                    {"role": "system", "content": "You are a terrain generator"},
                    {"role": "user", "content": caption},
                    {"role": "assistant", "content": None, "tool_calls": calls}
                ],
                "training_metadata": {
                    "sample_id": sample_id,
                    "validation_score": validation_result["score"],
                    "heightmap_hash": heightmap_hash,
                    "generation_time_ms": int(generation_time * 1000),
                    "terrain_template": features.get("terrain_type", "unknown"),
                    "feature_count": len(features.get("primary_features", [])) + len(features.get("secondary_features", [])),
                    "complexity_score": features.get("complexity", 0.5)
                }
            }
            
            # Update statistics
            self._update_sample_stats(features, validation_result["score"], generation_time)
            
            return sample
            
        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            return None
    
    def _hash_heightmap(self, heightmap: np.ndarray) -> str:
        """Create hash of heightmap for validation."""
        heightmap_bytes = heightmap.astype(np.float32).tobytes()
        return hashlib.sha256(heightmap_bytes).hexdigest()[:16]
    
    def _update_sample_stats(self, features: Dict, validation_score: float, generation_time: float):
        """Update generation statistics."""
        
        self.stats["total_generated"] += 1
        self.stats["total_valid"] += 1
        self.stats["generation_times"].append(generation_time)
        self.stats["validation_scores"].append(validation_score)
        
        # Track feature usage
        for feature in features.get("primary_features", []):
            self.stats["feature_usage"][feature] = self.stats["feature_usage"].get(feature, 0) + 1
        
        for feature in features.get("secondary_features", []):
            self.stats["feature_usage"][feature] = self.stats["feature_usage"].get(feature, 0) + 1
        
        # Track terrain types
        terrain_type = features.get("terrain_type", "unknown")
        self.stats["terrain_types"][terrain_type] = self.stats["terrain_types"].get(terrain_type, 0) + 1
    
    def generate_dataset(self, num_samples: int, batch_size: int = 100) -> str:
        """
        Generate complete dataset with validation.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Samples per batch (for memory management)
            
        Returns:
            Path to generated dataset manifest
        """
        
        print(f"Generating {num_samples} samples to {self.output_dir}")
        print(f"Mode: {self.mode}")
        
        # Generate in batches
        combined_path = self.output_dir / "dataset_all.jsonl"
        outfile = open(combined_path, 'w')
        
        successful_samples = 0
        
        with tqdm(total=num_samples, desc="Generating terrain") as pbar:
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                # Generate batch
                batch_samples = []
                for i in range(batch_start, batch_end):
                    sample = self.generate_sample(i)
                    if sample is not None:
                        batch_samples.append(sample)
                        successful_samples += 1
                    pbar.update(1)
                
                # Write samples to combined JSONL file
                for sample in batch_samples:
                    outfile.write(json.dumps(sample, separators=(',', ':')) + "\n")
        
        outfile.close()
        
        print(f"Successfully generated {successful_samples} valid samples out of {num_samples} attempts")
        print(f"Success rate: {successful_samples/num_samples*100:.1f}%")
        
        # Save complete manifest
        manifest = {
            "dataset_info": {
                "total_samples": successful_samples,
                "attempted_samples": num_samples,
                "tile_size": self.tile_size,
                "output_dir": str(self.output_dir),
                "generation_mode": self.mode,
                "generation_stats": self._finalize_stats()
            },
            "dataset_file": combined_path.name,
            "quality_metrics": self.quality_metrics.calculate_dataset_metrics(self.stats)
        }
        
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save detailed statistics
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Dataset manifest: {manifest_path}")
        print(f"Statistics: {stats_path}")
        print(f"Ready for training!")
        
        return str(manifest_path)
    
    def _finalize_stats(self) -> Dict[str, Any]:
        """Finalize and summarize generation statistics."""
        
        if not self.stats["generation_times"]:
            return self.stats
        
        return {
            **self.stats,
            "avg_generation_time": np.mean(self.stats["generation_times"]),
            "avg_validation_score": np.mean(self.stats["validation_scores"]),
            "success_rate": self.stats["total_valid"] / max(1, self.stats["total_generated"]),
            "most_common_features": sorted(
                self.stats["feature_usage"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "terrain_type_distribution": self.stats["terrain_types"]
        }


def main():
    """CLI entry point for enhanced dataset generation."""
    
    parser = argparse.ArgumentParser(description="Generate enhanced Text2Terrain training dataset")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=256, help="Terrain tile size")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="feature_based", 
                       choices=["feature_based", "legacy_compat"],
                       help="Generation mode")
    
    args = parser.parse_args()
    
    # Initialize enhanced generator
    generator = DatasetGeneratorV2(
        output_dir=args.output,
        tile_size=args.tile_size,
        seed=args.seed,
        mode=args.mode
    )
    
    # Generate dataset
    manifest_path = generator.generate_dataset(
        num_samples=args.n,
        batch_size=args.batch_size
    )
    
    print(f"\nEnhanced dataset generation complete!")
    print(f"Next steps:")
    print(f"  python -m src.data_gen.preprocessing {manifest_path}")
    print(f"  python -m src.training.train --data-path <preprocessed_output>")


if __name__ == "__main__":
    main()