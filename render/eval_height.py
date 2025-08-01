#!/usr/bin/env python3
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.procgen.core import TerrainEngine

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
    dataset_path = sys.argv[1]
    idx = int(sys.argv[2])
    RES = int(sys.argv[3])
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                record = json.loads(line)
                break
        else:
            sys.exit(1)
    tool_call = record["messages"][-1]["tool_calls"][0]
    args = json.loads(tool_call["function"]["arguments"])
    world_x = args["world_x"]
    world_y = args["world_y"]
    module_ids = args["module_ids"]
    parameters = args["parameters"]
    global_seed = args.get("global_seed", 42)
    seeds = args.get("seeds", [])
    if len(seeds) < len(module_ids):
        seeds.extend([(global_seed + i) & 0xFFFFFFFF for i in range(len(seeds), len(module_ids))])
    engine = TerrainEngine(tile_size=RES)
    heightmap = engine.generate_tile(world_x, world_y, module_ids, parameters, seeds, global_seed, use_fast=False)
    scale = parameters.get("height_scale", 1.0)
    heightmap = np.array((heightmap * scale).block_until_ready(), dtype=np.float32)
    sys.stdout.buffer.write(heightmap.tobytes())

if __name__ == "__main__":
    main()