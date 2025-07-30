"""
Data module for Text2Terrain training.

Efficient data loading with PyTorch DataLoader for training and validation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import json

from ..data_gen.preprocessing import load_processed_dataset


class TerrainDataset(Dataset):
    """
    PyTorch Dataset for Text2Terrain training data.
    
    Loads preprocessed .npz files with tokenized captions and normalized parameters.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to preprocessed .npz file
        """
        self.data_path = Path(data_path)
        self.data = load_processed_dataset(str(data_path))
        
        # Convert to tensors for faster access
        self.input_ids = torch.from_numpy(self.data["input_ids"]).long()
        self.attention_mask = torch.from_numpy(self.data["attention_mask"]).long()
        self.module_targets = torch.from_numpy(self.data["module_targets"]).float()
        self.param_targets = torch.from_numpy(self.data["param_targets"]).float()
        
        print(f"Loaded dataset: {len(self)} samples")
        print(f"  Input shape: {self.input_ids.shape}")
        print(f"  Module targets: {self.module_targets.shape}")
        print(f"  Parameter targets: {self.param_targets.shape}")
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "module_targets": self.module_targets[idx],
            "param_targets": self.param_targets[idx]
        }
    
    def get_sample_caption(self, idx: int) -> str:
        """Get original caption for a sample (for debugging)."""
        return self.data["captions"][idx]
    
    def get_parameter_names(self) -> list:
        """Get parameter names."""
        return list(self.data["param_names"])


class TerrainDataModule:
    """
    Data module for managing train/val data loading.
    
    Provides DataLoaders with consistent batching and shuffling.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize data module.
        
        Args:
            data_dir: Directory with processed train.npz and val.npz
            batch_size: Batch size for training
            num_workers: Number of DataLoader workers
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        
        print(f"Data module initialized:")
        print(f"  Data directory: {data_dir}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Pin memory: {pin_memory}")
    
    def setup(self):
        """Load datasets."""
        
        train_path = self.data_dir / "train.npz"
        val_path = self.data_dir / "val.npz"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        self.train_dataset = TerrainDataset(train_path)
        self.val_dataset = TerrainDataset(val_path)
        
        print(f"Datasets loaded:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        
        if self.train_dataset is None:
            self.setup()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Ensure consistent batch sizes
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        
        if self.val_dataset is None:
            self.setup()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def get_num_modules(self) -> int:
        """Get number of terrain modules."""
        return self.metadata["num_modules"]
    
    def get_num_parameters(self) -> int:
        """Get number of parameters."""
        return len(self.metadata["parameter_names"])
    
    def get_parameter_names(self) -> list:
        """Get parameter names."""
        return self.metadata["parameter_names"]
    
    def get_module_names(self) -> list:
        """Get module names."""
        return self.metadata["module_names"]
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return self.metadata["vocab_size"]


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles batching of variable-length sequences.
    """
    
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    module_targets = torch.stack([item["module_targets"] for item in batch])
    param_targets = torch.stack([item["param_targets"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "module_targets": module_targets,
        "param_targets": param_targets
    }


class TerrainDataModuleConfig:
    """Configuration class for data module."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
    
    def create_datamodule(self) -> TerrainDataModule:
        """Create data module from config."""
        
        return TerrainDataModule(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )