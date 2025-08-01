"""
Main dataset generator for Text2Terrain training data.

Generates synthetic datasets by:
1. Sampling parameter combinations
2. Generating terrain with procgen engine
3. Creating captions
4. Saving structured JSON traces
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

from tqdm import tqdm

from ..procgen import TerrainEngine, ModuleRegistry
from .captions import CaptionGenerator


class DatasetGenerator:
    """
    Generates synthetic training datasets for Text2Terrain.
    
    Creates varied terrain samples with natural language descriptions
    suitable for training neural text-to-parameter models.
    """
    
    def __init__(
        self, 
        output_dir: str,
        tile_size: int = 256,
        seed: int = 42,
        mode: str = "random"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_size = tile_size
        self.mode = mode
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Initialize components
        self.engine = TerrainEngine(tile_size=tile_size)
        self.caption_gen = CaptionGenerator(seed=seed)
        self.registry = self.engine.registry
        
        # Grid mode parameters
        if mode == "grid":
            self._setup_grid_mode()
        
        # Track generation statistics
        self.stats = {
            "total_generated": 0,
            "module_usage": {},
            "parameter_ranges": {}
        }
    
    def _setup_grid_mode(self):
        """Initialize grid-based parameter sampling."""
        self.param_grids = {
            "frequency": [0.0025, 0.01, 0.04],
            "persistence": [0.2, 0.5, 0.8], 
            "octaves": [2, 4, 6],
            "ridge_sharpness": [0.0, 0.5, 1.0],
            "erosion_speed": [0.0, 0.1, 0.3],
            "rain_amount": [0.3, 0.6, 1.0],
            "warp_amplitude": [0.0, 140.0, 280.0],
            "height_scale": [500.0, 1500.0, 3000.0]
        }
        
        # Generate all parameter combinations
        import itertools
        self.param_combinations = []
        keys = list(self.param_grids.keys())
        
        for combo in itertools.product(*[self.param_grids[k] for k in keys]):
            param_dict = dict(zip(keys, combo))
            self.param_combinations.append(param_dict)
        
        # Shuffle for variety while maintaining determinism
        self.rng.shuffle(self.param_combinations)
        self._grid_counter = 0
    
    def sample_parameters(self) -> Tuple[List[int], Dict[str, float]]:
        """
        Sample parameter configuration (random or grid-based).
        
        Returns:
            Tuple of (module_ids, parameters)
        """
        if self.mode == "grid":
            return self._sample_grid_parameters()
        else:
            return self._sample_random_parameters()
    
    def _sample_random_parameters(self) -> Tuple[List[int], Dict[str, float]]:
        """Original random parameter sampling."""
        # Sample 1-4 modules (more variety in combinations)
        available_modules = list(range(len(self.registry.list_modules())))
        num_modules = self.rng.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        
        # Ensure we always have at least one noise module
        noise_modules = [0, 1]  # perlin_noise, ridged_multi
        selected_modules = [self.rng.choice(noise_modules)]
        
        # Add additional modules
        other_modules = [m for m in available_modules if m not in selected_modules]
        if num_modules > 1 and other_modules:
            additional = self.rng.sample(other_modules, min(num_modules - 1, len(other_modules)))
            selected_modules.extend(additional)
        
        # Sort modules for consistent ordering (noise -> warp -> erosion)
        selected_modules.sort()
        
        # Sample parameters for selected modules
        all_params = self.registry.get_all_parameters()
        parameters = {}
        
        for param_name, (min_val, max_val, default) in all_params.items():
            # Use different sampling strategies for different parameter types
            if "frequency" in param_name:
                # Log-uniform for frequency (more interesting variation)
                log_min, log_max = np.log10(min_val), np.log10(max_val)
                log_val = self.np_rng.uniform(log_min, log_max)
                parameters[param_name] = 10 ** log_val
            elif "octaves" in param_name:
                # Discrete uniform for octaves
                parameters[param_name] = self.rng.randint(int(min_val), int(max_val))
            elif "iterations" in param_name:
                # Discrete uniform for iterations
                parameters[param_name] = self.rng.randint(int(min_val), int(max_val))
            else:
                # Uniform for most other parameters
                parameters[param_name] = self.np_rng.uniform(min_val, max_val)
        # Extra global height scale
        parameters["height_scale"] = self.np_rng.uniform(500.0, 3000.0)
        
        return selected_modules, parameters
    
    def _sample_grid_parameters(self) -> Tuple[List[int], Dict[str, float]]:
        """Sample from discrete parameter grid."""
        if self._grid_counter >= len(self.param_combinations):
            self._grid_counter = 0
        
        base_params = self.param_combinations[self._grid_counter].copy()
        self._grid_counter += 1
        
        # Determine module sequence based on parameters
        module_ids = self._select_modules_for_params(base_params)
        
        # Fill in defaults for unused parameters
        all_params = self.registry.get_all_parameters()
        parameters = {}
        for param_name, (min_val, max_val, default) in all_params.items():
            parameters[param_name] = base_params.get(param_name, default)
        
        return module_ids, parameters
    
    def _select_modules_for_params(self, params: Dict[str, float]) -> List[int]:
        """Select module sequence based on parameter characteristics."""
        modules = [0]  # Always start with perlin_noise
        
        # Add ridged noise for mountain terrain
        if params["ridge_sharpness"] > 0.1:
            modules.append(1)  # ridged_multi
        
        # Add domain warp for twisted terrain
        if params["warp_amplitude"] > 50.0:
            modules.append(2)  # domain_warp
        
        # Add erosion for weathered terrain
        if params["erosion_speed"] > 0.05:
            modules.append(3)  # hydraulic_erosion
        
        return modules
    
    def _get_terrain_archetype(self, modules: List[int], params: Dict[str, float]) -> str:
        """Determine terrain archetype name for consistent labeling."""
        has_ridged = 1 in modules
        has_warp = 2 in modules
        has_erosion = 3 in modules
        
        freq = params["frequency"]
        persistence = params["persistence"]
        
        if has_ridged and has_erosion:
            return "eroded_mountains"
        elif has_ridged:
            return "sharp_peaks"
        elif has_warp and has_erosion:
            return "twisted_valleys"
        elif has_warp:
            return "flowing_hills"
        elif has_erosion:
            return "weathered_plains"
        elif freq > 0.02:
            return "rolling_hills"
        elif persistence > 0.6:
            return "steep_terrain"
        else:
            return "gentle_plains"
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Generate a single training sample.
        
        Args:
            sample_id: Unique identifier for this sample
            
        Returns:
            Complete sample dictionary
        """
        
        # Sample configuration
        module_ids, parameters = self.sample_parameters()
        
        # Get terrain archetype for metadata
        archetype = self._get_terrain_archetype(module_ids, parameters) if self.mode == "grid" else None
        
        # Deterministic seeds derived from sample_id for reproducibility
        seeds = [((sample_id * 10007 + idx * 1013) % (2**31 - 1)) for idx in range(len(module_ids))]
        call_sequence = []
        for mid, seed in zip(module_ids, seeds):
            name = self.registry.get_module_name(mid)
            spec = self.registry.get_parameter_spec(mid)
            module_params = spec.extract_params(parameters)
            call_sequence.append({"function": name, "parameters": module_params, "seed": seed})
        
        # Generate world coordinates (vary across different regions)
        world_scale = 10  # Creates coordinates from -2,560 to +2,560
        world_x = self.rng.randint(-world_scale, world_scale) * self.tile_size
        world_y = self.rng.randint(-world_scale, world_scale) * self.tile_size
        
        # Prepare global generation parameters (tile will be generated later at inference/render time)
        global_seed = (sample_id * 48271) % (2**31 - 1)
        # Generate caption (deterministic in grid mode)
        caption = self.caption_gen.generate_caption(module_ids, parameters, archetype)
        
        # Function schema for dataset (Fireworks/OpenAI format)
        tool_schema = {
            "type": "function",
            "function": {
                "name": "generate_heightmap",
                "description": "Generate a terrain heightmap tile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tile_u": {"type": "integer"},
                        "tile_v": {"type": "integer"},
                        "world_x": {"type": "integer"},
                        "world_y": {"type": "integer"},
                        "global_seed": {"type": "integer"},
                        "module_ids": {"type": "array", "items": {"type": "integer"}},
                        "parameters": {"type": "object"},
                        "seeds": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["world_x", "world_y", "global_seed", "module_ids", "parameters"]
                }
            }
        }

        # Derive tile indices (integer grid) from world_x/world_y
        tile_u = world_x // self.tile_size
        tile_v = world_y // self.tile_size
        
        arguments_obj = {
            "tile_u": tile_u,
            "tile_v": tile_v,
            "world_x": world_x,
            "world_y": world_y,
            "global_seed": global_seed,
            "module_ids": module_ids,
            "parameters": parameters,
            "seeds": seeds
        }

        # Chat-style sample with tool calling format (EXACT Together.ai format)
        sample = {
            "messages": [
                {"role": "system", "content": "You are a terrain generator"},
                {"role": "user", "content": caption},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "generate_heightmap",
                                "arguments": json.dumps(arguments_obj)
                            }
                        }
                    ]
                }
            ]
        }
        
        # Add metadata for grid mode (doesn't affect training)
        if self.mode == "grid" and archetype:
            sample["metadata"] = {
                "archetype": archetype,
                "parameter_bins": {k: v for k, v in self.param_combinations[self._grid_counter-1].items()},
                "sample_id": sample_id,
                "tile_u": tile_u,
                "tile_v": tile_v
            }
        
        # Update statistics
        self._update_stats(module_ids, parameters)
        
        return sample
    
    def _update_stats(self, module_ids: List[int], parameters: Dict[str, float]):
        """Update generation statistics."""
        
        self.stats["total_generated"] += 1
        
        # Module usage
        for mid in module_ids:
            module_name = self.registry.get_module_name(mid)
            self.stats["module_usage"][module_name] = self.stats["module_usage"].get(module_name, 0) + 1
        
        # Parameter ranges
        for param_name, value in parameters.items():
            if param_name not in self.stats["parameter_ranges"]:
                self.stats["parameter_ranges"][param_name] = {"min": value, "max": value, "sum": 0, "count": 0}
            
            param_stats = self.stats["parameter_ranges"][param_name]
            param_stats["min"] = min(param_stats["min"], value)
            param_stats["max"] = max(param_stats["max"], value)
            param_stats["sum"] += value
            param_stats["count"] += 1
    
    def generate_dataset(self, num_samples: int, batch_size: int = 100) -> str:
        """
        Generate complete dataset.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Samples per batch (for memory management)
            
        Returns:
            Path to generated dataset manifest
        """
        
        print(f"Generating {num_samples} samples to {self.output_dir}")
        
        # Generate in batches
        combined_path = self.output_dir / "dataset_all.jsonl"
        outfile = open(combined_path, 'w')
        
        with tqdm(total=num_samples, desc="Generating terrain") as pbar:
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                # Generate batch
                batch_samples = []
                for i in range(batch_start, batch_end):
                    try:
                        sample = self.generate_sample(i)
                        batch_samples.append(sample)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error generating sample {i}: {e}")
                        continue
                
                # Write samples to combined JSONL file
                for sample in batch_samples:
                    outfile.write(json.dumps(sample, separators=(',', ':')) + "\n")
        
        outfile.close()
        # Save complete manifest
        manifest = {
            "dataset_info": {
                "total_samples": num_samples,
                "tile_size": self.tile_size,
                "output_dir": str(self.output_dir),
                "generation_stats": self.stats
            },
            "dataset_file": combined_path.name
        }
        
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save statistics
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Manifest saved to: {manifest_path}")
        print(f"Statistics saved to: {stats_path}")
        
        return str(manifest_path)


def main():
    """CLI entry point for dataset generation."""
    
    parser = argparse.ArgumentParser(description="Generate Text2Terrain training dataset")
    parser.add_argument("--n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=256, help="Terrain tile size")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"], 
                       help="Generation mode: 'random' for original stochastic, 'grid' for deterministic")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DatasetGenerator(
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
    
    print(f"\nDataset generation complete!")
    print(f"Run: python -m src.data_gen.preprocessing {manifest_path}")
    print("to prepare data for training.")


if __name__ == "__main__":
    main()