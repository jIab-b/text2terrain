"""
Preprocessing pipeline for training data.

Converts raw dataset to training-ready format:
1. Tokenize captions
2. Normalize parameters 
3. Create train/val splits
4. Save as efficient numpy arrays
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..procgen import ModuleRegistry


def preprocess_dataset(
    manifest_path: str,
    output_dir: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    val_split: float = 0.1,
    max_length: int = 128,
    seed: int = 42
) -> Dict[str, str]:
    """
    Preprocess raw dataset for training.
    
    Args:
        manifest_path: Path to dataset manifest JSON
        output_dir: Output directory for processed data
        model_name: HuggingFace model name for tokenizer
        val_split: Fraction of data for validation
        max_length: Maximum sequence length for tokenization
        seed: Random seed for train/val split
        
    Returns:
        Dictionary with paths to processed files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing dataset from {manifest_path}")
    print(f"Using tokenizer: {model_name}")
    
    # Load dataset manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    samples = manifest["samples"]
    print(f"Processing {len(samples)} samples")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize module registry for parameter normalization
    registry = ModuleRegistry()
    # Recreate registry (this should match the one used in generation)
    from ..procgen.core import TerrainEngine
    temp_engine = TerrainEngine()
    registry = temp_engine.registry
    
    # Process samples
    processed_samples = []
    
    print("Tokenizing captions and normalizing parameters...")
    for sample in tqdm(samples):
        try:
            processed = process_single_sample(sample, tokenizer, registry, max_length)
            processed_samples.append(processed)
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    # Train/validation split
    train_samples, val_samples = train_test_split(
        processed_samples, 
        test_size=val_split,
        random_state=seed,
        shuffle=True
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Convert to numpy arrays and save
    train_path = save_processed_split(train_samples, output_path / "train.npz", tokenizer)
    val_path = save_processed_split(val_samples, output_path / "val.npz", tokenizer)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "max_length": max_length,
        "vocab_size": tokenizer.vocab_size,
        "num_modules": len(registry.list_modules()),
        "module_names": [name for _, name in registry.list_modules()],
        "parameter_names": list(registry.get_all_parameters().keys()),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "original_manifest": manifest_path
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = output_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"\nPreprocessing complete!")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Tokenizer: {tokenizer_path}")
    
    return {
        "train": str(train_path),
        "val": str(val_path),
        "metadata": str(metadata_path),
        "tokenizer": str(tokenizer_path)
    }


def process_single_sample(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    registry: ModuleRegistry,
    max_length: int
) -> Dict[str, Any]:
    """Process a single sample for training."""
    
    # Tokenize caption
    caption = sample["caption"]
    tokens = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    # Create module target (multi-hot encoding)
    module_ids = sample["module_ids"]
    num_modules = len(registry.list_modules())
    module_target = np.zeros(num_modules, dtype=np.float32)
    for mid in module_ids:
        if 0 <= mid < num_modules:
            module_target[mid] = 1.0
    
    # Normalize parameters
    parameters = sample["parameters"]
    all_params = registry.get_all_parameters()
    
    # Create parameter vector (normalized to [0, 1])
    param_vector = []
    param_names = []
    
    for param_name, (min_val, max_val, default) in all_params.items():
        if param_name in parameters:
            value = parameters[param_name]
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            # Use default if parameter not present
            normalized = (default - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0.0, 1.0)
        
        param_vector.append(normalized)
        param_names.append(param_name)
    
    param_vector = np.array(param_vector, dtype=np.float32)
    
    return {
        "input_ids": tokens["input_ids"].flatten(),
        "attention_mask": tokens["attention_mask"].flatten(),
        "module_target": module_target,
        "param_target": param_vector,
        "param_names": param_names,
        "original_caption": caption,
        "sample_id": sample["id"]
    }


def save_processed_split(
    samples: List[Dict[str, Any]],
    output_path: Path,
    tokenizer: AutoTokenizer
) -> str:
    """Save processed samples as numpy arrays."""
    
    if not samples:
        print(f"Warning: No samples to save to {output_path}")
        return str(output_path)
    
    # Stack all samples
    input_ids = np.stack([s["input_ids"] for s in samples])
    attention_masks = np.stack([s["attention_mask"] for s in samples])
    module_targets = np.stack([s["module_target"] for s in samples])
    param_targets = np.stack([s["param_target"] for s in samples])
    
    # Store sample metadata
    captions = [s["original_caption"] for s in samples]
    sample_ids = [s["sample_id"] for s in samples]
    
    # Save as compressed numpy arrays
    np.savez_compressed(
        output_path,
        input_ids=input_ids,
        attention_mask=attention_masks,
        module_targets=module_targets,
        param_targets=param_targets,
        captions=captions,
        sample_ids=sample_ids,
        param_names=samples[0]["param_names"]  # Same for all samples
    )
    
    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Module targets shape: {module_targets.shape}")
    print(f"  Parameter targets shape: {param_targets.shape}")
    
    return str(output_path)


def load_processed_dataset(data_path: str) -> Dict[str, np.ndarray]:
    """Load preprocessed dataset from .npz file."""
    
    data = np.load(data_path)
    return {
        "input_ids": data["input_ids"],
        "attention_mask": data["attention_mask"], 
        "module_targets": data["module_targets"],
        "param_targets": data["param_targets"],
        "captions": data["captions"],
        "sample_ids": data["sample_ids"],
        "param_names": data["param_names"]
    }


def main():
    """CLI entry point for preprocessing."""
    
    parser = argparse.ArgumentParser(description="Preprocess Text2Terrain dataset")
    parser.add_argument("manifest_path", help="Path to dataset manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", 
                       help="HuggingFace model for tokenizer")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Preprocess dataset
    output_paths = preprocess_dataset(
        manifest_path=args.manifest_path,
        output_dir=args.output,
        model_name=args.model,
        val_split=args.val_split,
        max_length=args.max_length,
        seed=args.seed
    )
    
    print("\nDataset preprocessing complete!")
    print("Ready for training with:")
    print(f"  python -m src.training.train --data-path {args.output}")


if __name__ == "__main__":
    main()