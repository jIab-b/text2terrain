import os
from openai import OpenAI
from typing import Dict, Any

class LLMCaptionGenerator:

    def __init__(self, base_generator, model_name: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.base = base_generator
        self.model = model_name

        self.client = OpenAI(
            api_key=api_key,
        )

    def generate_from_analysis(
        self,
        terrain_analysis: Dict[str, Any],
        features: Dict[str, Any],
        parameters: Dict[str, float]
    ) -> str:
        # Generate base caption
        short_caption = self.base.generate_from_analysis(terrain_analysis, features, parameters)
        # Prompt LLM to expand and vary the caption
        prompt = (
            f"Paraphrase and expand this terrain caption with more descriptive detail: '{short_caption}'"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful terrain caption assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()