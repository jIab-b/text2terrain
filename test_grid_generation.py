#!/usr/bin/env python3
"""
Quick test of grid-based deterministic generation.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_gen.generator import DatasetGenerator

def main():
    print("🧪 Testing deterministic grid mode generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create generator in grid mode
        generator = DatasetGenerator(
            output_dir=temp_dir,
            tile_size=256,
            seed=42,
            mode="grid"
        )
        
        print(f"📊 Grid has {len(generator.param_combinations)} parameter combinations")
        
        # Generate a few samples
        print("\n🎯 Sample captions generated:")
        for i in range(min(5, len(generator.param_combinations))):
            sample = generator.generate_sample(i)
            caption = sample["messages"][1]["content"]
            archetype = sample.get("metadata", {}).get("archetype", "N/A")
            print(f"  {i}: '{caption}' [{archetype}]")
        
        # Test determinism
        print("\n🔄 Testing determinism...")
        sample1 = generator.generate_sample(0)
        generator._grid_counter = 0  # Reset to test same sample
        sample2 = generator.generate_sample(0)
        
        caption1 = sample1["messages"][1]["content"]
        caption2 = sample2["messages"][1]["content"]
        
        if caption1 == caption2:
            print("✅ Same sample_id produces identical caption")
        else:
            print(f"❌ Caption mismatch: '{caption1}' vs '{caption2}'")
        
        # Show format compatibility
        print(f"\n📋 Together.ai compatible format:")
        print(f"  Messages: {len(sample1['messages'])}")
        print(f"  System: '{sample1['messages'][0]['content']}'")
        print(f"  User: '{sample1['messages'][1]['content']}'")
        print(f"  Assistant has tool_calls: {'tool_calls' in sample1['messages'][2]}")
        
        print("\n🎉 Test completed!")

if __name__ == "__main__":
    main()