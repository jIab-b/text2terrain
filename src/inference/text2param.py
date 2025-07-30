"""
Text-to-parameter prediction for Text2Terrain inference.

Lightweight wrapper for the trained LoRA model that converts
natural language descriptions to terrain generation parameters.
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from ..training.model import Text2TerrainModel
from ..procgen import ModuleRegistry


class Text2ParamPredictor:
    """
    Text-to-parameter prediction interface.
    
    Loads trained LoRA model and provides clean interface for
    converting text descriptions to terrain parameters.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "auto",
        module_threshold: float = 0.5
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained LoRA model weights
            tokenizer_path: Path to tokenizer (if None, uses same dir as model)
            device: Device to run inference on ("auto", "cpu", "cuda")
            module_threshold: Threshold for module selection
        """
        
        self.model_path = Path(model_path)
        self.module_threshold = module_threshold
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing Text2Terrain predictor on {self.device}")
        
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = self.model_path.parent / "tokenizer"
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = Text2TerrainModel.load_lora_weights(str(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize parameter registry for denormalization
        from ..procgen.core import TerrainEngine
        temp_engine = TerrainEngine()
        self.registry = temp_engine.registry
        self.all_params = self.registry.get_all_parameters()
        self.param_names = list(self.all_params.keys())
        
        print(f"Model loaded successfully:")
        print(f"  Modules: {self.model.num_modules}")
        print(f"  Parameters: {self.model.num_parameters}")
        print(f"  Parameter names: {self.param_names}")
    
    def predict(
        self,
        text: str,
        max_length: int = 128,
        return_probs: bool = False
    ) -> Dict:
        """
        Predict terrain parameters from text description.
        
        Args:
            text: Natural language terrain description
            max_length: Maximum tokenization length
            return_probs: Whether to return module probabilities
            
        Returns:
            Dictionary with module IDs, parameters, and optionally probabilities
        """
        
        # Tokenize input
        tokens = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            selected_modules, normalized_params = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                module_threshold=self.module_threshold
            )
        
        # Convert to CPU numpy
        selected_modules = selected_modules.cpu().numpy().flatten()
        normalized_params = normalized_params.cpu().numpy().flatten()
        
        # Get selected module IDs
        module_ids = [i for i, selected in enumerate(selected_modules) if selected > 0]
        
        # Denormalize parameters
        parameters = self._denormalize_parameters(normalized_params)
        
        result = {
            "text": text,
            "module_ids": module_ids,
            "module_names": [self.registry.get_module_name(mid) for mid in module_ids],
            "parameters": parameters,
            "normalized_parameters": dict(zip(self.param_names, normalized_params.tolist()))
        }
        
        if return_probs:
            # Get module probabilities
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                module_probs = outputs["module_probs"].cpu().numpy().flatten()
            
            result["module_probabilities"] = dict(enumerate(module_probs.tolist()))
        
        return result
    
    def _denormalize_parameters(self, normalized_params: torch.Tensor) -> Dict[str, float]:
        """Convert normalized [0,1] parameters back to original ranges."""
        
        parameters = {}
        
        for i, (param_name, (min_val, max_val, default)) in enumerate(self.all_params.items()):
            if i < len(normalized_params):
                normalized = float(normalized_params[i])
                # Denormalize from [0, 1] to [min_val, max_val]
                value = min_val + normalized * (max_val - min_val)
                parameters[param_name] = value
            else:
                parameters[param_name] = default
        
        return parameters
    
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict parameters for a batch of texts.
        
        Args:
            texts: List of terrain descriptions
            max_length: Maximum tokenization length
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            tokens = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            
            # Predict batch
            with torch.no_grad():
                selected_modules, normalized_params = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    module_threshold=self.module_threshold
                )
            
            # Process each sample in batch
            selected_modules = selected_modules.cpu().numpy()
            normalized_params = normalized_params.cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                module_ids = [k for k, selected in enumerate(selected_modules[j]) if selected > 0]
                parameters = self._denormalize_parameters(normalized_params[j])
                
                result = {
                    "text": text,
                    "module_ids": module_ids,
                    "module_names": [self.registry.get_module_name(mid) for mid in module_ids],
                    "parameters": parameters
                }
                results.append(result)
        
        return results
    
    def get_example_predictions(self) -> List[Dict]:
        """Get predictions for example terrain descriptions."""
        
        examples = [
            "jagged mountain peaks with deep valleys",
            "rolling grassy hills",
            "weathered rocky cliffs with erosion channels",
            "smooth desert dunes",
            "volcanic landscape with rough terrain",
            "alpine ridges with glacial carving",
            "gentle meadow with soft undulations"
        ]
        
        return self.predict_batch(examples)
    
    def save_config(self, config_path: str):
        """Save predictor configuration."""
        
        config = {
            "model_path": str(self.model_path),
            "device": self.device,
            "module_threshold": self.module_threshold,
            "num_modules": self.model.num_modules,
            "num_parameters": self.model.num_parameters,
            "parameter_names": self.param_names
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, config_path: str):
        """Load predictor from configuration file."""
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            model_path=config["model_path"],
            device=config["device"],
            module_threshold=config["module_threshold"]
        )