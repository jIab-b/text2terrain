#!/usr/bin/env python3
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.procgen import TerrainEngine
except ImportError:
    try:
        from src.engine.terrain_composer import TerrainComposer as TerrainEngine
    except ImportError:
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        sys.exit(1)
    dataset_path = sys.argv[1]
    idx = int(sys.argv[2])
    RES = int(sys.argv[3])
    temp_file = sys.argv[4]
    
    try:
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i == idx:
                    record = json.loads(line)
                    break
            else:
                sys.exit(1)
        
        tool_calls = record["messages"][-1]["tool_calls"]
        heightmap_calls = [call for call in tool_calls if call["function"]["name"] == "generate_heightmap"]
        
        if not heightmap_calls:
            sys.exit(1)
        
        tool_call = heightmap_calls[-1]
        args = json.loads(tool_call["function"]["arguments"])
        
        world_x = args.get("world_x", 0)
        world_y = args.get("world_y", 0)
        module_ids = args.get("module_ids", [0])
        parameters = args.get("parameters", {})
        global_seed = args.get("global_seed", 42)
        seeds = args.get("seeds", [])
        
        if len(seeds) < len(module_ids):
            seeds.extend([(global_seed + i) & 0xFFFFFFFF for i in range(len(seeds), len(module_ids))])
        
        engine = TerrainEngine(tile_size=RES)
        heightmap = engine.generate_heightmap(
            world_x=world_x,
            world_y=world_y,
            module_ids=module_ids,
            parameters=parameters,
            seeds=seeds,
            global_seed=global_seed,
            legacy_mode=True
        )
        
        if hasattr(heightmap, 'block_until_ready'):
            heightmap = heightmap.block_until_ready()
        heightmap = np.array(heightmap, dtype=np.float32)
        
        if heightmap.shape != (RES, RES):
            sys.exit(1)
        
        with open(temp_file, 'wb') as f:
            f.write(heightmap.tobytes())
        
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()