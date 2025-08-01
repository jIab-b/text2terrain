#!/usr/bin/env python3
"""
Validation script to test deterministic data generation.
"""

import json
import tempfile
from pathlib import Path
from generator import DatasetGenerator

def test_determinism():
    """Test that grid mode generates deterministic, one-to-one mappings."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test grid mode
        generator = DatasetGenerator(
            output_dir=temp_dir,
            tile_size=256,
            seed=42,
            mode="grid"
        )
        
        print(f"Grid mode has {len(generator.param_combinations)} parameter combinations")
        
        # Generate same sample twice
        sample1 = generator.generate_sample(0)
        generator._grid_counter = 0  # Reset counter
        sample2 = generator.generate_sample(0)
        
        # Check determinism
        caption1 = sample1["messages"][1]["content"]
        caption2 = sample2["messages"][1]["content"]
        
        print(f"Sample 0 caption: '{caption1}'")
        print(f"Sample 0 repeated: '{caption2}'")
        
        assert caption1 == caption2, "Captions should be identical for same sample_id"
        print("✅ Single sample determinism test passed")
        
        # Test caption uniqueness across samples
        caption_to_params = {}
        duplicates = 0
        
        for i in range(min(100, len(generator.param_combinations))):
            sample = generator.generate_sample(i)
            caption = sample["messages"][1]["content"]
            
            # Extract parameters from tool call
            args_str = sample["messages"][2]["tool_calls"][0]["function"]["arguments"]
            args_obj = json.loads(args_str)
            param_signature = tuple(sorted(args_obj["parameters"].items()))
            
            if caption in caption_to_params:
                if caption_to_params[caption] != param_signature:
                    print(f"❌ Caption collision: '{caption}' maps to different parameters")
                    duplicates += 1
                else:
                    print(f"✅ Same caption maps to same parameters: '{caption}'")
            else:
                caption_to_params[caption] = param_signature
        
        print(f"\n📊 Results:")
        print(f"Tested {min(100, len(generator.param_combinations))} samples")
        print(f"Found {len(caption_to_params)} unique captions")
        print(f"Duplicates with different parameters: {duplicates}")
        
        if duplicates == 0:
            print("✅ All captions map to unique parameter sets!")
        else:
            print(f"❌ Found {duplicates} caption collisions")
        
        # Test some specific examples
        print(f"\n🔍 Sample captions:")
        for i in range(min(5, len(caption_to_params))):
            sample = generator.generate_sample(i)
            caption = sample["messages"][1]["content"]
            archetype = sample.get("metadata", {}).get("archetype", "N/A")
            print(f"  {i}: '{caption}' ({archetype})")

def test_together_ai_format():
    """Test that output format is compatible with Together.ai."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = DatasetGenerator(
            output_dir=temp_dir,
            tile_size=256,
            seed=42,
            mode="grid"
        )
        
        sample = generator.generate_sample(0)
        
        # Check required fields
        assert "messages" in sample, "Missing 'messages' field"
        assert len(sample["messages"]) == 3, "Should have 3 messages (system, user, assistant)"
        
        messages = sample["messages"]
        assert messages[0]["role"] == "system", "First message should be system"
        assert messages[1]["role"] == "user", "Second message should be user"
        assert messages[2]["role"] == "assistant", "Third message should be assistant"
        
        # Check tool call format
        assert "tool_calls" in messages[2], "Assistant message should have tool_calls"
        tool_call = messages[2]["tool_calls"][0]
        assert tool_call["type"] == "function", "Tool call should be function type"
        assert tool_call["function"]["name"] == "generate_heightmap", "Function name should match"
        
        # Check arguments are valid JSON
        args_str = tool_call["function"]["arguments"]
        args_obj = json.loads(args_str)
        required_keys = ["world_x", "world_y", "global_seed", "module_ids", "parameters", "seeds"]
        for key in required_keys:
            assert key in args_obj, f"Missing required argument: {key}"
        
        print("✅ Together.ai format validation passed")
        
        # Write sample to check with together CLI
        sample_file = Path(temp_dir) / "test_sample.jsonl"
        with open(sample_file, 'w') as f:
            f.write(json.dumps(sample) + "\n")
        
        print(f"✅ Sample written to {sample_file}")
        print("   You can test with: together files check <file>")

if __name__ == "__main__":
    print("🧪 Testing deterministic data generation...")
    test_determinism()
    print("\n📋 Testing Together.ai format compatibility...")
    test_together_ai_format()
    print("\n🎉 All tests completed!")