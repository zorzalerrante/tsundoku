from urllib.parse import urlparse

import pandas as pd
import scattertext
from cytoolz import sliding_window
from emoji.unicode_codes import EMOJI_UNICODE
from gensim.utils import deaccent as gensim_deaccent
from nltk.tokenize.casual import casual_tokenize
from aves.features.twokenize import tokenize as ark_twokenize
from tsundoku.features.re import PUNCTUATION_RE, URL_RE


def score_frequency_table(gg_df, alpha_w=0.001, top_k=15):
    """
    param @gg_df: DataFrame with documents as rows and terms as columns.
    """
    scorer = scattertext.LogOddsRatioUninformativeDirichletPrior(alpha_w=alpha_w)

    top_words = {}

    for group_i, idx in enumerate(gg_df.index, start=1):
        positive = gg_df.T[idx]
        negative = gg_df.T.drop(idx, axis=1).sum(axis=1)
        group_scores = scorer.get_scores(positive, negative)
        top_words[
            idx
        ] = group_scores  # .sort_values(ascending=False).head(top_k).index.values

    return pd.DataFrame(top_words)


EMOJI_VALUES = set(EMOJI_UNICODE["es"].values())


def tokenize(
    text,
    remove_urls=True,
    deaccent=False,
    remove_punctuation=True,
    lower=True,
    ngram_range=None,
    stopwords=None,
    use_nltk=False
):
    if deaccent:
        text = gensim_deaccent(text)

    if lower:
        text = text.lower()

    if not use_nltk:
        tokens = ark_twokenize(text)
    else:
        tokens = casual_tokenize(text)

    results = []

    if remove_punctuation:
        tokens = filter(lambda x: PUNCTUATION_RE.match(x) is None, tokens)

    if remove_urls:
        tokens = filter(lambda x: URL_RE.match(x) is None, tokens)

    tokens = list(filter(lambda x: x, tokens))

    results.extend(tokens)

    if ngram_range is not None:
        for i in range(*ngram_range):

            if len(tokens) < i:
                continue

            for composite in sliding_window(i, tokens):
                composite = list(composite)

                # do not allow composite emojis
                if any(x in EMOJI_VALUES for x in composite):
                    continue

                # do not allow composite mentions/hashtags
                if any(x.startswith("@") for x in composite) or any(
                    x.startswith("#") for x in composite
                ):
                    continue

                if remove_urls:
                    if any(URL_RE.match(x) for x in composite):
                        continue

                if remove_punctuation:
                    if any(PUNCTUATION_RE.match(x) for x in composite):
                        continue

                if stopwords:
                    if all(x in stopwords for x in composite):
                        continue

                    if composite[-1] in stopwords:
                        continue

                results.append(" ".join(composite))

    if stopwords:
        results = list(filter(lambda x: not x in stopwords, results))

    return results
