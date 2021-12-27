from urllib.parse import urlparse

import pandas as pd
import scattertext
from aves.features.twokenize import tokenize as twokenize
from cytoolz import sliding_window
from emoji.unicode_codes import EMOJI_UNICODE
from gensim.utils import deaccent as gensim_deaccent

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
):
    if lower:
        text = text.lower()

    if deaccent:
        text = gensim_deaccent(text)

    tokens = list(twokenize(text))

    results = []

    if remove_punctuation:
        tokens = filter(lambda x: PUNCTUATION_RE.match(x) is None, tokens)

    if remove_urls:
        tokens = filter(lambda x: URL_RE.match(x) is None, tokens)

    tokens = list(tokens)

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


from collections import defaultdict
from itertools import chain

from cytoolz import frequencies, keyfilter, keymap, merge_with, valmap
from lru import LRU


def word_frequencies(text_iter, tokenize):
    return frequencies(chain(*map(tokenize, text_iter)))


def merge_frequencies(*freqs):
    return merge_with(sum, *freqs)


def get_tweet_terms(tweet, tokenize):
    if "retweeted_status" in tweet:
        rt = tweet["retweeted_status"]
        text = tweet_text(rt)
    else:
        text = tweet_text(tweet)

    terms = word_frequencies((text,), tokenize)

    urls = tweet_urls(tweet)

    if urls:
        terms.update({url: 1 for url in urls})

    return terms


def cached_tokenizer(tokenize, cache_size=512):
    seen = LRU(cache_size)

    def get_terms(tweet):
        tweet_id = tweet["id"]

        if "retweeted_status" in tweet:
            tweet_rt_id = tweet["retweeted_status"]["id"]
            if tweet_rt_id in seen:
                term_counts = seen[tweet_rt_id]
            else:
                term_counts = get_tweet_terms(tweet["retweeted_status"], tokenize)
                seen[tweet_rt_id] = term_counts
        else:
            if tweet_id in seen:
                term_counts = seen[tweet_id]
            else:
                term_counts = get_tweet_terms(tweet, tokenize)
                seen[tweet_id] = term_counts

        return term_counts

    return get_terms


def build_tokenizer(ngram_range=None, stopwords=None, cache_size=512):
    tokenize_fn = tokenize_function()
    tokenize = lambda x: tokenize_fn(x, ngram_range=ngram_range, stopwords=stopwords)
    return cached_tokenizer(tokenize, cache_size=cache_size)
