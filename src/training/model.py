"""
Text2Terrain LoRA model implementation.

Adds specialized heads to a base language model for:
- Module selection (multi-label classification)
- Parameter regression (normalized to [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Optional, Tuple


class Text2TerrainModel(nn.Module):
    """
    Text-to-terrain parameter prediction model.
    
    Architecture:
    - Base: Mistral-7B with LoRA adapters
    - Module head: Multi-label classification for terrain modules
    - Parameter head: Regression for normalized parameters [0,1]
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        num_modules: int = 4,
        num_parameters: int = 16,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        hidden_dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modules = num_modules
        self.num_parameters = num_parameters
        
        # Load base model
        config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply LoRA to base model
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Get hidden size from config
        hidden_size = config.hidden_size
        
        # Prediction heads
        self.module_head = nn.Sequential(
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_modules)
        )
        
        self.parameter_head = nn.Sequential(
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_parameters),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Initialize heads
        self._init_heads()
    
    def _init_heads(self):
        """Initialize prediction heads with sensible weights."""
        
        for module in [self.module_head, self.parameter_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        module_targets: Optional[torch.Tensor] = None,
        param_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            module_targets: Module targets for training [batch_size, num_modules]
            param_targets: Parameter targets for training [batch_size, num_parameters]
            
        Returns:
            Dictionary with logits, predictions, and losses
        """
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state, mean-pooled across sequence
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden_states = hidden_states * mask_expanded
        pooled = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        
        # Predictions
        module_logits = self.module_head(pooled)  # [batch_size, num_modules]
        param_preds = self.parameter_head(pooled)  # [batch_size, num_parameters]
        
        # Module probabilities (sigmoid for multi-label)
        module_probs = torch.sigmoid(module_logits)
        
        result = {
            "module_logits": module_logits,
            "module_probs": module_probs,
            "param_preds": param_preds,
            "pooled_features": pooled
        }
        
        # Compute losses if targets provided
        if module_targets is not None:
            module_loss = F.binary_cross_entropy_with_logits(
                module_logits, module_targets.float()
            )
            result["module_loss"] = module_loss
        
        if param_targets is not None:
            param_loss = F.mse_loss(param_preds, param_targets.float())
            result["param_loss"] = param_loss
        
        # Combined loss
        if module_targets is not None and param_targets is not None:
            # Weight losses (parameters typically need more emphasis)
            total_loss = result["module_loss"] + 5.0 * result["param_loss"]
            result["loss"] = total_loss
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        module_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for inference.
        
        Args:
            input_ids: Tokenized text
            attention_mask: Attention mask
            module_threshold: Threshold for module selection
            
        Returns:
            Tuple of (selected_modules, parameters)
        """
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Module selection (threshold-based)
            module_probs = outputs["module_probs"]
            selected_modules = (module_probs > module_threshold).long()
            
            # Parameters are already in [0, 1]
            parameters = outputs["param_preds"]
            
            return selected_modules, parameters
    
    def get_lora_parameters(self):
        """Get LoRA parameters for saving/loading."""
        return {k: v for k, v in self.named_parameters() if "lora" in k}
    
    def freeze_base_model(self):
        """Freeze base model parameters (keep only LoRA + heads trainable)."""
        
        for name, param in self.base_model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        
        # Ensure heads remain trainable
        for param in self.module_head.parameters():
            param.requires_grad = True
        for param in self.parameter_head.parameters():
            param.requires_grad = True
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights and prediction heads."""
        
        state_dict = {
            "lora_state_dict": self.base_model.state_dict(),
            "module_head_state_dict": self.module_head.state_dict(),
            "parameter_head_state_dict": self.parameter_head.state_dict(),
            "config": {
                "num_modules": self.num_modules,
                "num_parameters": self.num_parameters,
                "model_name": getattr(self, 'model_name', 'mistralai/Mistral-7B-Instruct-v0.1')
            }
        }
        
        torch.save(state_dict, path)
    
    @classmethod
    def load_lora_weights(cls, path: str, model_name: str = None):
        """Load model with LoRA weights."""
        
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint["config"]
        
        if model_name is None:
            model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.1")
        
        # Create model
        model = cls(
            model_name=model_name,
            num_modules=config["num_modules"],
            num_parameters=config["num_parameters"]
        )
        
        # Load weights
        model.base_model.load_state_dict(checkpoint["lora_state_dict"])
        model.module_head.load_state_dict(checkpoint["module_head_state_dict"])
        model.parameter_head.load_state_dict(checkpoint["parameter_head_state_dict"])
        
        return model