"""
Generate natural language captions for terrain parameters.

Maps parameter combinations to descriptive text using rule-based templates.
"""

import random
from typing import Dict, List, Any
import numpy as np


class CaptionGenerator:
    """
    Generates natural language descriptions for terrain parameters.
    
    Uses template-based approach with parameter ranges to create
    varied, descriptive captions that match the terrain characteristics.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._setup_templates()
    
    def _setup_templates(self):
        """Initialize caption templates and vocabulary."""
        
        # Base terrain types
        self.terrain_types = {
            "mountain": ["peaks", "ridges", "slopes", "summits", "crests"],
            "valley": ["valleys", "ravines", "gorges", "canyons", "depressions"],
            "hills": ["hills", "mounds", "knolls", "rises", "undulations"],
            "plains": ["plains", "flats", "steppes", "meadows", "fields"]
        }
        
        # Descriptive adjectives by parameter ranges
        self.frequency_words = {
            "low": ["broad", "sweeping", "vast", "wide", "expansive"],
            "medium": ["rolling", "moderate", "gentle", "undulating"],
            "high": ["sharp", "jagged", "detailed", "fine", "intricate"]
        }
        
        self.amplitude_words = {
            "low": ["subtle", "gentle", "mild", "soft", "low"],
            "medium": ["moderate", "noticeable", "prominent"],
            "high": ["dramatic", "steep", "towering", "extreme", "massive"]
        }
        
        self.erosion_words = {
            "none": [],
            "light": ["weathered", "aged", "worn"],
            "heavy": ["eroded", "carved", "sculpted", "deeply cut", "heavily weathered"]
        }
        
        self.warp_words = {
            "none": [],
            "light": ["twisted", "curved", "flowing"],
            "heavy": ["distorted", "warped", "contorted", "folded"]
        }
        
        # Environmental descriptors
        self.environments = [
            "alpine", "arctic", "desert", "tropical", "temperate",
            "volcanic", "coastal", "grassland", "forest", "tundra"
        ]
        
        # Weather/atmosphere
        self.atmosphere = [
            "misty", "foggy", "windswept", "sunlit", "shadowy",
            "stormy", "peaceful", "harsh", "pristine", "rugged"
        ]
        
        # Geological terms
        self.geological = [
            "granite", "limestone", "volcanic", "sedimentary", "rocky",
            "sandstone", "basalt", "quartzite", "shale", "marble"
        ]
    
    def generate_caption(
        self, 
        module_ids: List[int], 
        parameters: Dict[str, float],
        module_names: List[str] = None
    ) -> str:
        """
        Generate a natural language caption for given parameters.
        
        Args:
            module_ids: List of terrain module IDs
            parameters: Parameter values
            module_names: Optional module names (for debugging)
            
        Returns:
            Natural language description
        """
        
        # Determine primary terrain characteristics
        freq = parameters.get("frequency", 0.01)
        amplitude = self._estimate_amplitude(parameters)
        has_erosion = any("erosion" in str(mid) for mid in module_ids)
        has_warp = any("warp" in str(mid) for mid in module_ids)
        has_ridged = any("ridged" in str(mid) for mid in module_ids)
        
        # Build caption components
        components = []
        
        # Atmospheric descriptor (30% chance)
        if self.rng.random() < 0.3:
            components.append(self.rng.choice(self.atmosphere))
        
        # Geological descriptor (40% chance)
        if self.rng.random() < 0.4:
            components.append(self.rng.choice(self.geological))
        
        # Primary terrain feature
        if has_ridged or amplitude > 0.7:
            terrain_type = "mountain"
        elif freq > 0.05:
            terrain_type = "hills"
        elif amplitude < 0.3:
            terrain_type = "plains"
        else:
            terrain_type = self.rng.choice(["valley", "hills"])
        
        # Frequency-based descriptors
        if freq < 0.005:
            freq_desc = self.rng.choice(self.frequency_words["low"])
        elif freq > 0.02:
            freq_desc = self.rng.choice(self.frequency_words["high"])
        else:
            freq_desc = self.rng.choice(self.frequency_words["medium"])
        
        # Amplitude-based descriptors
        if amplitude < 0.3:
            amp_desc = self.rng.choice(self.amplitude_words["low"])
        elif amplitude > 0.7:
            amp_desc = self.rng.choice(self.amplitude_words["high"])
        else:
            amp_desc = self.rng.choice(self.amplitude_words["medium"])
        
        # Combine frequency and amplitude descriptors (avoid redundancy)
        if freq_desc != amp_desc:
            components.extend([freq_desc, amp_desc])
        else:
            components.append(freq_desc)
        
        # Main terrain feature
        components.append(self.rng.choice(self.terrain_types[terrain_type]))
        
        # Erosion effects
        if has_erosion:
            erosion_strength = parameters.get("rain_amount", 0.5) + parameters.get("erosion_speed", 0.1)
            if erosion_strength > 0.6:
                erosion_desc = self.rng.choice(self.erosion_words["heavy"])
            else:
                erosion_desc = self.rng.choice(self.erosion_words["light"])
            
            if erosion_desc and self.rng.random() < 0.7:
                components.append("with")
                components.append(erosion_desc)
                if self.rng.random() < 0.5:
                    components.append(self.rng.choice(["valleys", "channels", "gullies"]))
        
        # Warping effects
        if has_warp:
            warp_strength = parameters.get("warp_amplitude", 100)
            if warp_strength > 200:
                warp_desc = self.rng.choice(self.warp_words["heavy"])
            else:
                warp_desc = self.rng.choice(self.warp_words["light"])
            
            if warp_desc and self.rng.random() < 0.6:
                if "with" not in components:
                    components.append("with")
                components.append(warp_desc)
                components.append("formations")
        
        # Environmental context (20% chance)
        if self.rng.random() < 0.2:
            env = self.rng.choice(self.environments)
            if env not in components:
                components.insert(-1 if len(components) > 1 else 0, env)
        
        return " ".join(components)
    
    def _estimate_amplitude(self, parameters: Dict[str, float]) -> float:
        """Estimate overall terrain amplitude from parameters."""
        
        amplitude = 0.0
        
        # Base noise amplitude
        persistence = parameters.get("persistence", 0.5)
        octaves = parameters.get("octaves", 4)
        amplitude += persistence * octaves / 8.0
        
        # Ridge sharpness increases perceived amplitude
        ridge_sharpness = parameters.get("ridge_sharpness", 0.0)
        amplitude += ridge_sharpness * 0.3
        
        # Erosion can increase or decrease amplitude
        erosion_strength = parameters.get("erosion_speed", 0.0)
        if erosion_strength > 0.15:
            amplitude += 0.2  # Deep valleys increase amplitude
        
        return min(amplitude, 1.0)
    
    def generate_batch(
        self, 
        param_list: List[Dict], 
        module_ids_list: List[List[int]]
    ) -> List[str]:
        """Generate captions for a batch of parameter sets."""
        
        captions = []
        for params, module_ids in zip(param_list, module_ids_list):
            caption = self.generate_caption(module_ids, params)
            captions.append(caption)
        
        return captions


# Predefined templates for specific terrain types
TERRAIN_TEMPLATES = {
    "mountain": [
        "{adj1} {adj2} mountain {feature} with {detail}",
        "{atmosphere} {geological} peaks and {feature}",
        "{adj1} alpine {feature} {detail}"
    ],
    "valley": [
        "{adj1} {feature} carved by {process}",
        "deep {geological} {feature} with {detail}",
        "{atmosphere} river {feature} and {detail}"
    ],
    "plateau": [
        "{adj1} {geological} plateau with {detail}",
        "elevated {feature} and {adj2} {detail}",
        "{atmosphere} mesa formation with {process}"
    ]
}