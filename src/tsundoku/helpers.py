import gzip

import rapidjson as json
import toml


def read_file(filename, encoding="utf-8"):
    with open(filename, "rt", encoding=encoding) as f:
        return f.read().strip()


def read_list(filename, encoding="utf-8"):
    return read_file(filename, encoding=encoding).split("\n")


def read_json(filename):
    filename = str(filename)
    if filename.endswith("gz"):
        fn = gzip.open
    else:
        fn = open

    with fn(filename, "rt") as f:
        return json.load(f)


def read_toml(filename):
    filename = str(filename)
    if filename.endswith("gz"):
        fn = gzip.open
    else:
        fn = open

    with fn(filename, "rt") as f:
        return toml.load(f)


def write_json(obj, filename):
    filename = str(filename)
    if filename.endswith("gz"):
        fn = gzip.open
    else:
        fn = open

    with fn(filename, "wt") as f:
        json.dump(obj, f)
