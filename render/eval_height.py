import sys,json,struct,random
from pathlib import Path
import numpy as np
from src.procgen.core import TerrainEngine

def get_line(path,index):
    with open(path,'r') as f:
        for i,line in enumerate(f):
            if i==index:
                return line
    raise IndexError

def parse_args(line):
    rec=json.loads(line)
    arg_str=None
    for m in rec["messages"]:
        if "tool_calls" in m:
            arg_str=m["tool_calls"][0]["function"]["arguments"]
            break
    return json.loads(arg_str)

def main():
    path=sys.argv[1]
    idx=int(sys.argv[2])
    res=int(sys.argv[3])
    line=get_line(path,idx)
    args=parse_args(line)
    eng=TerrainEngine(tile_size=res)
    hm=eng.generate_tile(args["world_x"],args["world_y"],args["module_ids"],args["parameters"],args["seeds"],args["global_seed"])
    arr=np.array(hm,dtype=np.float32).flatten()
    sys.stdout.buffer.write(arr.tobytes())

if __name__=="__main__":
    main()