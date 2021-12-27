from glob import glob

import dask
import numpy as np
import pandas as pd

from tsundoku.features.dates import date_from_filename
from tsundoku.features.re import PUNCTUATION_RE
from tsundoku.helpers import read_list


def to_array(x):
    return np.squeeze(np.array(x))


def build_elem_to_id(dask_df, key, keep="last"):
    return (
        dask_df.drop_duplicates(subset=key, keep=keep)[[key]]
        .compute()
        .assign(row_id=lambda x: range(len(x)))
        .set_index(key)
    )


def remove_stopwords(df, stopwords_file, token_column="token"):
    stopwords = set(read_list(stopwords_file))
    df = df[(~df[token_column].isin(stopwords))]
    return df


def remove_punctuation(df, token_column="token"):
    return df[(~df[token_column].str.contains(pat=PUNCTUATION_RE))]


def filter_vocabulary(
    vocabulary,
    min_freq=50,
    stopwords_file=None,
    remove_punctuation=True,
    token_column="token",
    frequency_column="frequency",
    remove_twitter_urls=True,
):
    if remove_punctuation:
        vocabulary = vocabulary[
            (~vocabulary[token_column].str.contains(pat=PUNCTUATION_RE))
        ]

    if stopwords_file is not None:
        stopwords = set(read_list(stopwords_file))
        vocabulary = vocabulary[(~vocabulary[token_column].isin(stopwords))]

    if remove_twitter_urls:
        vocabulary = vocabulary[
            ~vocabulary[token_column].str.startswith("https://t.co")
        ]

    return (
        vocabulary.groupby(token_column)
        .sum()
        .sort_values(frequency_column, ascending=False)
        .pipe(lambda x: x[x[frequency_column] >= min_freq])
        .assign(token_id=lambda x: range(len(x)))
    )


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
