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
from PIL import Image
import base64
import io
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
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_size = tile_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Initialize components
        self.engine = TerrainEngine(tile_size=tile_size)
        self.caption_gen = CaptionGenerator(seed=seed)
        self.registry = self.engine.registry
        
        # Track generation statistics
        self.stats = {
            "total_generated": 0,
            "module_usage": {},
            "parameter_ranges": {}
        }
    
    def sample_parameters(self) -> Tuple[List[int], Dict[str, float], List[int]]:
        """
        Sample random parameter configuration.
        
        Returns:
            Tuple of (module_ids, parameters, seeds)
        """
        
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
        
        # Generate per-module seeds
        seeds = [self.rng.randint(0, 2**31 - 1) for _ in selected_modules]
        
        return selected_modules, parameters, seeds
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Generate a single training sample.
        
        Args:
            sample_id: Unique identifier for this sample
            
        Returns:
            Complete sample dictionary
        """
        
        # Sample configuration
        module_ids, parameters, seeds = self.sample_parameters()
        
        # Generate world coordinates (vary across different regions)
        world_scale = 10  # Creates coordinates from -2,560 to +2,560
        world_x = self.rng.randint(-world_scale, world_scale) * self.tile_size
        world_y = self.rng.randint(-world_scale, world_scale) * self.tile_size
        
        # Generate terrain
        global_seed = self.rng.randint(0, 2**31 - 1)
        heightmap = self.engine.generate_tile(
            world_x=world_x,
            world_y=world_y,
            module_ids=module_ids,
            parameters=parameters,
            seeds=seeds,
            global_seed=global_seed
        )
        
        # Convert to numpy and normalize to [0, 1]
        heightmap_np = np.array(heightmap)
        heightmap_np = (heightmap_np - heightmap_np.min()) / (heightmap_np.max() - heightmap_np.min() + 1e-8)
        
        # Convert to 16-bit PNG for storage
        heightmap_16bit = (heightmap_np * 65535).astype(np.uint16)
        
        # Generate caption
        caption = self.caption_gen.generate_caption(module_ids, parameters)
        
        # Create sample dictionary
        sample = {
            "id": sample_id,
            "caption": caption,
            "module_ids": module_ids,
            "parameters": parameters,
            "seeds": seeds,
            "global_seed": global_seed,
            "tile_origin": [world_x, world_y],
            "tile_size": self.tile_size,
            "heightmap_shape": list(heightmap_np.shape),
            "heightmap_range": [float(heightmap_np.min()), float(heightmap_np.max())],
        }
        
        # Save heightmap as PNG file
        heightmap_path = self.output_dir / f"heightmap_{sample_id:06d}.png"
        Image.fromarray(heightmap_16bit, mode='I;16').save(heightmap_path)
        sample["heightmap_file"] = str(heightmap_path.name)
        
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
        samples = []
        
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
                
                samples.extend(batch_samples)
                
                # Save batch to disk
                batch_file = self.output_dir / f"batch_{batch_start:06d}_{batch_end:06d}.json"
                with open(batch_file, 'w') as f:
                    json.dump(batch_samples, f, indent=2)
        
        # Save complete manifest
        manifest = {
            "dataset_info": {
                "total_samples": len(samples),
                "tile_size": self.tile_size,
                "output_dir": str(self.output_dir),
                "generation_stats": self.stats
            },
            "samples": samples
        }
        
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save statistics
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {len(samples)}")
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
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DatasetGenerator(
        output_dir=args.output,
        tile_size=args.tile_size,
        seed=args.seed
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