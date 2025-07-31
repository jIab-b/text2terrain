import sys, json, numpy as np
from pathlib import Path
from importlib import import_module

sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root

from src.procgen import TerrainEngine


def main():
    if len(sys.argv) != 3:
        print("Usage: python reconstruct_heightmap.py <sample_json> <output_png>")
        sys.exit(1)
    sample_path = Path(sys.argv[1])
    output_png = Path(sys.argv[2])

    data = json.loads(Path(sample_path).read_text())
    sample = data[0] if isinstance(data, list) else data

    engine = TerrainEngine(tile_size=sample["tile_size"])

    heightmap = engine.generate_tile(
        world_x=sample["tile_origin"][0],
        world_y=sample["tile_origin"][1],
        module_ids=sample["module_ids"],
        parameters=sample["parameters"],
        seeds=[((sample["id"] * 10007 + idx * 1013) % (2**31 - 1)) for idx in range(len(sample["module_ids"]))],
        global_seed=(sample["id"] * 48271) % (2**31 - 1),
    )

    import PIL.Image as Image

    hm_np = np.array(heightmap)
    hm_norm = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
    hm_8 = (hm_norm * 255).astype(np.uint8)
    Image.fromarray(hm_8, mode="L").save(output_png)
    print(f"Saved heightmap: {hm_8.shape}, min={hm_np.min():.3f}, max={hm_np.max():.3f}")


if __name__ == "__main__":
    main()
