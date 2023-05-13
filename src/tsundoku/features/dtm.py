from collections import Counter, defaultdict

import logging
import numpy as np
from scipy.sparse import dok_matrix
import pandas as pd
import dask.dataframe as dd


def build_vocabulary(vocab_df, token_column, to_lower=True, chunksize=10000):
    if type(vocab_df) == pd.core.frame.DataFrame:
        vocab_df = dd.from_pandas(vocab_df, chunksize=chunksize)

    return (
        vocab_df[[token_column]]
        .explode(token_column)
        .assign(
            token=lambda x: x[token_column].str.lower() if to_lower else x[token_column]
        )["token"]
        .value_counts()
        .rename("frequency")
        .to_frame()
        .compute()
        .sort_values("frequency", ascending=False)
        .reset_index()
        .rename(columns={"index": "token"})
    )


def tokens_to_document_term_matrix(
        df, id_column, token_column, vocabulary, id_to_row=None):
    if type(vocabulary) == pd.core.series.Series:
        vocab_name = vocabulary.name
        token_to_column = (
            vocabulary.reset_index().set_index(vocab_name)["index"].to_dict()
        )
    else:
        token_to_column = dict(vocabulary)

    token_counts = defaultdict(Counter)

    if id_to_row is None:
        id_to_row = dict(zip(df[id_column], range(len(df))))

    for i, (elem_id, elem_tokens) in enumerate(df[[id_column, token_column]].values):
        if elem_id in id_to_row:
            row_id = id_to_row[elem_id]
        else:
            continue

        elem_tokens = filter(lambda x: x in token_to_column, elem_tokens)

        token_counts[row_id].update(elem_tokens)
    dtm = dok_matrix(
        (max(id_to_row.values()) + 1, max(token_to_column.values()) + 1), dtype=int
    )

    for row_id, elem_token_counts in token_counts.items():
        for token, token_count in elem_token_counts.items():
            column_id = token_to_column[token]
            dtm[row_id, column_id] = token_count

    return dtm.tocsr()
