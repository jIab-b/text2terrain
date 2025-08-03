"""
Test script for new architecture compatibility and functionality.

Verifies that the new terrain generation system works correctly
and maintains compatibility with existing JSON format.
"""

import json
import numpy as np
from pathlib import Path

from src.engine import TerrainComposer, HeightmapAnalyzer
from src.data_gen import DatasetGeneratorV2
from src.compatibility import LegacyAdapter, JSONValidator


def test_terrain_composer():
    """Test basic terrain generation with new composer."""
    print("Testing TerrainComposer...")
    
    composer = TerrainComposer(tile_size=64)  # Small size for testing
    
    # Test legacy mode (for compatibility)
    heightmap_legacy = composer.generate_heightmap(
        world_x=0, world_y=0,
        module_ids=[0, 1],  # perlin + ridged
        parameters={
            "frequency": 0.02,
            "octaves": 4,
            "persistence": 0.5,
            "height_scale": 1000.0,
            "ridge_sharpness": 0.8
        },
        seeds=[42, 43],
        legacy_mode=True
    )
    
    print(f"Legacy heightmap shape: {heightmap_legacy.shape}")
    print(f"Legacy heightmap range: {heightmap_legacy.min():.2f} to {heightmap_legacy.max():.2f}")
    
    # Test feature-based mode
    features = {
        "terrain_type": "mountains",
        "primary_features": ["mountain_peaks"],
        "secondary_features": ["ridges"],
        "biome": "alpine",
        "complexity": 0.7
    }
    
    parameters = {
        "mountain_height": 0.8,
        "mountain_steepness": 0.7,
        "ridge_prominence": 0.6,
        "base_frequency": 0.01,
        "height_scale": 1000.0
    }
    
    heightmap_features = composer.generate_heightmap(
        world_x=0, world_y=0,
        features=features,
        parameters=parameters,
        seeds=[42, 43, 44],
        legacy_mode=False
    )
    
    print(f"Feature-based heightmap shape: {heightmap_features.shape}")
    print(f"Feature-based heightmap range: {heightmap_features.min():.2f} to {heightmap_features.max():.2f}")
    
    return True


def test_heightmap_analyzer():
    """Test terrain analysis functionality."""
    print("\nTesting HeightmapAnalyzer...")
    
    # Create simple test heightmap
    heightmap = np.zeros((64, 64))
    
    # Add a mountain peak
    center = 32
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            heightmap[i, j] = max(0, 100 - dist * 3)
    
    analyzer = HeightmapAnalyzer(tile_size=64)
    analysis = analyzer.analyze(heightmap)
    
    print(f"Detected terrain type: {analysis['terrain_classification']['primary_type']}")
    print(f"Peaks detected: {analysis['feature_detection']['peaks_detected']}")
    print(f"Elevation range: {analysis['elevation_stats']['range']:.2f}")
    
    return analysis['feature_detection']['peaks_detected'] > 0


def test_json_compatibility():
    """Test JSON format compatibility with renderer."""
    print("\nTesting JSON compatibility...")
    
    validator = JSONValidator()
    adapter = LegacyAdapter()
    
    # Create test data
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
    
    legacy_data = adapter.convert_to_legacy_format(
        features, parameters, 
        world_x=0, world_y=0, 
        seeds=[42, 43], 
        global_seed=42,
        tile_size=256
    )
    
    print(f"Legacy data: {json.dumps(legacy_data, indent=2)}")
    
    # Validate legacy format
    is_valid, errors = validator.validate_legacy_format(legacy_data)
    print(f"Legacy format valid: {is_valid}")
    if errors:
        print(f"Validation errors: {errors}")
    
    return is_valid


def test_data_generation():
    """Test enhanced data generation pipeline."""
    print("\nTesting enhanced data generation...")
    
    output_dir = Path("test_output")
    
    generator = DatasetGeneratorV2(
        output_dir=str(output_dir),
        tile_size=64,  # Small for testing
        seed=42
    )
    
    # Generate a single sample
    sample = generator.generate_sample(0)
    
    if sample is None:
        print("Sample generation failed!")
        return False
    
    print(f"Generated sample with validation score: {sample['training_metadata']['validation_score']:.3f}")
    print(f"Caption: {sample['messages'][1]['content']}")
    
    # Validate the JSON structure
    tool_call = sample['messages'][2]['tool_calls'][0]
    args_str = tool_call['function']['arguments']
    args_data = json.loads(args_str)
    
    validator = JSONValidator()
    is_valid, errors = validator.validate_enhanced_format(args_data)
    
    print(f"Enhanced JSON valid: {is_valid}")
    if errors:
        print(f"Validation errors: {errors}")
    
    # Clean up
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    return is_valid and sample['training_metadata']['validation_score'] > 0.5


def main():
    """Run all tests."""
    print("Testing new Text2Terrain architecture...")
    print("=" * 50)
    
    tests = [
        ("TerrainComposer", test_terrain_composer),
        ("HeightmapAnalyzer", test_heightmap_analyzer), 
        ("JSON Compatibility", test_json_compatibility),
        ("Data Generation", test_data_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"{test_name}: ✗ ERROR - {e}")
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! New architecture is working correctly.")
        print("\nNext steps:")
        print("1. Generate dataset: python -m src.data_gen.generator_v2 --output data/samples --n 100")
        print("2. Test with renderer: use existing render system with generated JSON")
    else:
        print("❌ Some tests failed. Check the errors above.")
        for test_name, result, error in results:
            if not result:
                print(f"  - {test_name}: {error or 'Failed'}")


if __name__ == "__main__":
    main()