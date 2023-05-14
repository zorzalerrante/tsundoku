import gzip
from glob import glob
import rapidjson as json
import toml
import dask
import pandas as pd

from tsundoku.utils.dates import date_from_filename


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


def process_daily_files(path, fname_function, pipe_function=None, add_date=True):
    files = glob(str(path))
    tasks = [dask.delayed(fname_function)(fname) for fname in files]
    results = dask.compute(*tasks)
    df = pd.DataFrame.from_records(results)

    if pipe_function is not None:
        df = df.pipe(pipe_function)

    if add_date:
        dates = pd.Series(list(map(date_from_filename, files)), name="date")
        return df.set_index(pd.to_datetime(dates)).sort_index()
    else:
        return df
