from tsundoku.utils.re import PUNCTUATION_RE
from tsundoku.utils.files import read_list


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


def build_elem_to_id(dask_df, key, keep="last"):
    return (
        dask_df.drop_duplicates(subset=key, keep=keep)[[key]]
        .compute()
        .assign(row_id=lambda x: range(len(x)))
        .set_index(key)
    )
