import argparse, json, gzip, pathlib


def open_file(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def convert(in_path, out_path):
    data = []
    with open_file(in_path) as f_in:
        for line in f_in:
            if line.strip():
                data.append(json.loads(line))
    with open(out_path, "w") as f_out:
        json.dump(data, f_out, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    args = p.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
