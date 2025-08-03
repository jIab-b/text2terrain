"""
Simple test for core functionality without complex dependencies.
"""

import json
import numpy as np

def test_legacy_adapter_simple():
    """Test just the legacy adapter with correct tile_size."""
    print("Testing LegacyAdapter with tile_size...")
    
    try:
        from src.compatibility import LegacyAdapter, JSONValidator
        
        adapter = LegacyAdapter()
        validator = JSONValidator()
        
        features = {
            "terrain_type": "mountains",
            "primary_features": ["steep_peaks"],
            "complexity": 0.8
        }
        
        parameters = {
            "mountain_height": 0.9,
            "mountain_steepness": 0.8,
            "height_scale": 1500.0
        }
        
        # Test with tile_size = 64 (like in the main test)
        world_x, world_y = 0, 0  # Use simple coordinates
        legacy_data = adapter.convert_to_legacy_format(
            features, parameters, 
            world_x, world_y, 
            seeds=[42, 43], 
            global_seed=42,
            tile_size=64  # This should fix the issue
        )
        
        print(f"Generated legacy data: {json.dumps(legacy_data, indent=2)}")
        
        # Validate
        is_valid, errors = validator.validate_legacy_format(legacy_data)
        print(f"Validation result: {is_valid}")
        if errors:
            print(f"Errors: {errors}")
        
        return is_valid
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_terrain_composer_basic():
    """Test basic terrain generation without scipy dependencies."""
    print("\nTesting basic TerrainComposer...")
    
    try:
        from src.engine import TerrainComposer
        
        composer = TerrainComposer(tile_size=64)
        
        # Test legacy mode only (avoid scipy dependencies)
        heightmap = composer.generate_heightmap(
            world_x=0, world_y=0,
            module_ids=[0],  # Just perlin noise
            parameters={
                "frequency": 0.02,
                "octaves": 4,
                "persistence": 0.5,
                "height_scale": 1000.0
            },
            seeds=[42],
            legacy_mode=True
        )
        
        print(f"Heightmap shape: {heightmap.shape}")
        print(f"Heightmap range: {heightmap.min():.2f} to {heightmap.max():.2f}")
        
        # Check if values are reasonable
        is_valid = (
            heightmap.shape == (64, 64) and
            heightmap.min() >= 0 and
            heightmap.max() > 0 and
            not np.any(np.isnan(heightmap))
        )
        
        print(f"Basic terrain generation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        return is_valid
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run simple tests."""
    print("Simple Text2Terrain Architecture Test")
    print("=" * 40)
    
    tests = [
        ("LegacyAdapter", test_legacy_adapter_simple),
        ("TerrainComposer", test_terrain_composer_basic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"{test_name}: ✗ ERROR - {e}")
    
    print("\n" + "=" * 40)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Simple tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ Core functionality working!")
    else:
        print("✗ Some core issues remain")


if __name__ == "__main__":
    main()