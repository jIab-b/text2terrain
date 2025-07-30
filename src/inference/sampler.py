"""
Terrain sampler that combines text-to-parameter prediction with terrain generation.

This is the main interface for generating terrain from text descriptions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import hashlib

from .text2param import Text2ParamPredictor
from ..procgen import TerrainEngine


class TerrainSampler:
    """
    Complete text-to-terrain generation pipeline.
    
    Combines text-to-parameter prediction with procedural terrain generation
    to create heightmaps from natural language descriptions.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        tile_size: int = 256,
        device: str = "auto"
    ):
        """
        Initialize terrain sampler.
        
        Args:
            model_path: Path to trained LoRA model
            tokenizer_path: Path to tokenizer (optional)
            tile_size: Size of generated terrain tiles
            device: Device for model inference
        """
        
        # Initialize text-to-parameter predictor
        self.predictor = Text2ParamPredictor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        
        # Initialize terrain engine
        self.engine = TerrainEngine(tile_size=tile_size)
        self.tile_size = tile_size
        
        print(f"TerrainSampler initialized:")
        print(f"  Tile size: {tile_size}")
        print(f"  Device: {self.predictor.device}")
    
    def generate_terrain(
        self,
        text: str,
        world_x: int = 0,
        world_y: int = 0,
        global_seed: int = 42,
        use_text_seed: bool = True
    ) -> Dict:
        """
        Generate terrain from text description.
        
        Args:
            text: Natural language terrain description
            world_x: World X coordinate
            world_y: World Y coordinate
            global_seed: Global random seed
            use_text_seed: Whether to derive seeds from text for consistency
            
        Returns:
            Dictionary with heightmap, parameters, and metadata
        """
        
        # Predict parameters from text
        prediction = self.predictor.predict(text)
        module_ids = prediction["module_ids"]
        parameters = prediction["parameters"]
        
        # Generate per-module seeds
        if use_text_seed:
            # Deterministic seeds based on text content
            seeds = self._generate_text_seeds(text, len(module_ids))
        else:
            # Use global seed with offsets
            seeds = [global_seed + i for i in range(len(module_ids))]
        
        # Generate terrain
        heightmap = self.engine.generate_tile(
            world_x=world_x,
            world_y=world_y,
            module_ids=module_ids,
            parameters=parameters,
            seeds=seeds,
            global_seed=global_seed
        )
        
        # Convert to numpy array
        heightmap_np = np.array(heightmap, dtype=np.float32)
        
        return {
            "text": text,
            "heightmap": heightmap_np,
            "module_ids": module_ids,
            "module_names": prediction["module_names"],
            "parameters": parameters,
            "seeds": seeds,
            "world_coordinates": [world_x, world_y],
            "tile_size": self.tile_size,
            "heightmap_stats": {
                "min": float(heightmap_np.min()),
                "max": float(heightmap_np.max()),
                "mean": float(heightmap_np.mean()),
                "std": float(heightmap_np.std())
            }
        }
    
    def generate_tile_grid(
        self,
        text: str,
        grid_size: int = 3,
        center_x: int = 0,
        center_y: int = 0,
        global_seed: int = 42
    ) -> Dict:
        """
        Generate a grid of terrain tiles for seamless world exploration.
        
        Args:
            text: Terrain description
            grid_size: Size of tile grid (e.g., 3 = 3x3 grid)
            center_x: Center tile X coordinate
            center_y: Center tile Y coordinate
            global_seed: Global seed for consistency
            
        Returns:
            Dictionary with grid of heightmaps and metadata
        """
        
        # Predict parameters once for entire grid
        prediction = self.predictor.predict(text)
        module_ids = prediction["module_ids"]
        parameters = prediction["parameters"]
        
        # Generate seeds from text for consistency
        seeds = self._generate_text_seeds(text, len(module_ids))
        
        # Generate grid of tiles
        tiles = {}
        heightmaps = {}
        
        half_grid = grid_size // 2
        
        for dy in range(-half_grid, half_grid + 1):
            for dx in range(-half_grid, half_grid + 1):
                tile_x = center_x + dx
                tile_y = center_y + dy
                
                world_x = tile_x * self.tile_size
                world_y = tile_y * self.tile_size
                
                # Generate tile
                heightmap = self.engine.generate_tile(
                    world_x=world_x,
                    world_y=world_y,
                    module_ids=module_ids,
                    parameters=parameters,
                    seeds=seeds,
                    global_seed=global_seed
                )
                
                heightmap_np = np.array(heightmap, dtype=np.float32)
                
                tiles[(tile_x, tile_y)] = {
                    "heightmap": heightmap_np,
                    "world_coordinates": [world_x, world_y]
                }
                heightmaps[(tile_x, tile_y)] = heightmap_np
        
        return {
            "text": text,
            "grid_size": grid_size,
            "center_coordinates": [center_x, center_y],
            "tiles": tiles,
            "heightmaps": heightmaps,  # Convenient access to just heightmaps
            "module_ids": module_ids,
            "module_names": prediction["module_names"],
            "parameters": parameters,
            "seeds": seeds,
            "tile_size": self.tile_size
        }
    
    def generate_with_variations(
        self,
        text: str,
        num_variations: int = 4,
        world_x: int = 0,
        world_y: int = 0,
        seed_offset: int = 0
    ) -> List[Dict]:
        """
        Generate multiple terrain variations from the same text.
        
        Args:
            text: Terrain description
            num_variations: Number of variations to generate
            world_x: World X coordinate
            world_y: World Y coordinate
            seed_offset: Starting seed offset
            
        Returns:
            List of terrain generation results
        """
        
        variations = []
        
        for i in range(num_variations):
            result = self.generate_terrain(
                text=text,
                world_x=world_x,
                world_y=world_y,
                global_seed=seed_offset + i,
                use_text_seed=False  # Use different seeds for variation
            )
            result["variation_id"] = i
            variations.append(result)
        
        return variations
    
    def _generate_text_seeds(self, text: str, num_seeds: int) -> List[int]:
        """Generate deterministic seeds from text content."""
        
        # Create hash from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Generate seeds from hash
        seeds = []
        for i in range(num_seeds):
            seed_str = f"{text_hash}_{i}"
            seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
            # Convert first 8 hex chars to int
            seed = int(seed_hash[:8], 16) % (2**31 - 1)
            seeds.append(seed)
        
        return seeds
    
    def save_heightmap(
        self,
        heightmap: np.ndarray,
        output_path: str,
        format: str = "png"
    ):
        """
        Save heightmap to file.
        
        Args:
            heightmap: Height data array
            output_path: Output file path
            format: File format ("png", "npy", "tiff")
        """
        
        output_path = Path(output_path)
        
        if format == "png":
            from PIL import Image
            # Normalize to [0, 65535] for 16-bit PNG
            normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
            heightmap_16bit = (normalized * 65535).astype(np.uint16)
            Image.fromarray(heightmap_16bit, mode='I;16').save(output_path)
            
        elif format == "npy":
            np.save(output_path, heightmap)
            
        elif format == "tiff":
            from PIL import Image
            # Save as 32-bit float TIFF
            Image.fromarray(heightmap, mode='F').save(output_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_terrain_statistics(self, heightmap: np.ndarray) -> Dict:
        """Calculate terrain statistics."""
        
        return {
            "shape": heightmap.shape,
            "min_height": float(heightmap.min()),
            "max_height": float(heightmap.max()),
            "mean_height": float(heightmap.mean()),
            "std_height": float(heightmap.std()),
            "height_range": float(heightmap.max() - heightmap.min()),
            "roughness": float(np.mean(np.abs(np.gradient(heightmap)[0])) + np.mean(np.abs(np.gradient(heightmap)[1]))),
        }
    
    def demo_generation(self, save_dir: str = None) -> List[Dict]:
        """
        Generate demo terrains with various descriptions.
        
        Args:
            save_dir: Optional directory to save heightmaps
            
        Returns:
            List of generated terrain results
        """
        
        demo_texts = [
            "rugged mountain peaks with snow caps",
            "rolling green hills with gentle slopes", 
            "desert dunes with wind patterns",
            "volcanic landscape with rough lava flows",
            "eroded canyon with steep walls",
            "alpine meadow with scattered boulders",
            "coastal cliffs with weathered surfaces"
        ]
        
        results = []
        
        for i, text in enumerate(demo_texts):
            print(f"Generating: {text}")
            
            result = self.generate_terrain(
                text=text,
                world_x=i * 1000,  # Spread out in world space
                world_y=0,
                global_seed=42
            )
            
            # Save heightmap if directory provided
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"terrain_{i:02d}_{text.replace(' ', '_')[:20]}.png"
                self.save_heightmap(
                    result["heightmap"], 
                    save_path / filename,
                    format="png"
                )
                result["saved_path"] = str(save_path / filename)
            
            results.append(result)
        
        return results