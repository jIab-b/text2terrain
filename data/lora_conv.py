import argparse, json, gzip, pathlib, sys

def open_file(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def convert(in_path, out_path):
    with open_file(in_path) as f_in, open_file(out_path, "wt") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                sys.stderr.write("Skipping line without messages key\n")
                continue
            msgs = []
            for m in obj["messages"]:
                m = dict(m)
                if m.get("content") is None:
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        tc = m["tool_calls"][0]["function"]["arguments"]
                        m["content"] = tc
                    else:
                        m["content"] = ""
                msgs.append(m)
            f_out.write(json.dumps({"messages": msgs}, separators=(",", ":")) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    args = p.parse_args()
    convert(args.input, args.output)

if __name__ == "__main__":
    main()
