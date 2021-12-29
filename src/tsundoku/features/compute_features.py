# -*- coding: utf-8 -*-
import logging
import os
import sys
from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path

import click
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from tsundoku.features.tweets import TWEET_DTYPES
from tsundoku.features.urls import DISCARD_URLS, get_domain
from tsundoku.helpers import read_toml, write_json


@click.command()
@click.option("--start_at", type=str, default="")
def main(start_at):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(Path(os.environ['TSUNDOKU_PROJECT_PATH']) / "config.toml")["project"]
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config["environment"].get("n_jobs", 2))))

    source_path = Path(config["path"]["data"]) / "raw" / "json"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_folders = sorted(glob(str(source_path / "*")))
    logging.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    for tweet_path in source_folders:
        date = str(os.path.basename(tweet_path))

        if start_at and date < start_at:
            continue

        target = Path(config["path"]["data"]) / "interim" / f"{date}"

        if not target.exists():
            target.mkdir(parents=True)
            logging.info(f"created: {tweet_path} -> {target}")
        else:
            logging.info(
                f"{target} already exists. skipping"
            )
            continue

        tweets = dd.read_json(
            Path(tweet_path) / "*.gz", meta=TWEET_DTYPES, dtype=TWEET_DTYPES
        )

        if tweets.npartitions <= 0:
            logger.warning(f"{date} has no files")
            continue

        logger.info(f"{date} -> computing user metrics")
        compute_user_metrics(tweets, target)
        logger.info(f"{date} -> computing tweet metrics")
        compute_tweet_metrics(tweets, target)
        logger.info(f"{date} -> done! :D")


def compute_user_metrics(tweets, target_path):
    users = None

    if not (target_path / "unique_users.json.gz").exists():
        users = (
            tweets.drop_duplicates(subset="user.id")
            .compute()
            .filter(regex="^user\.?.*")
        )
        users.to_json(
            target_path / "unique_users.json.gz",
            compression="gzip",
            orient="records",
            date_format="iso",
            lines=True,
        )

    if not (target_path / "user_name_vocabulary.json.gz").exists():
        if users is None:
            users = pd.read_json(target_path / "unique_users.json.gz", lines=True)

        (
            users[["user.id", "user.name_tokens"]]
            .explode("user.name_tokens")
            .rename(columns={"user.name_tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .groupby("token")
            .size()
            .rename("frequency")
            .sort_values()
            .to_frame()
            .reset_index()
            .to_json(
                target_path / "user_name_vocabulary.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "user_description_vocabulary.json.gz").exists():
        if users is None:
            users = pd.read_json(target_path / "unique_users.json.gz", lines=True)

        (
            users[["user.id", "user.description_tokens"]]
            .explode("user.description_tokens")
            .rename(columns={"user.description_tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .groupby("token")
            .size()
            .rename("frequency")
            .sort_values()
            .to_frame()
            .reset_index()
            .to_json(
                target_path / "user_description_vocabulary.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )


def compute_tweet_metrics(tweets, target_path):

    if not (target_path / "tweets_per_user.json.gz").exists():
        (
            tweets.drop_duplicates("id")
            .groupby("user.id")
            .size()
            .compute()
            .reset_index()
            .to_json(
                target_path / "tweets_per_user.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    tweet_vocabulary = None

    if not (target_path / "tweet_vocabulary.json.gz").exists():
        tweet_vocabulary = (
            tweets.drop_duplicates("id")
            .explode("tweet.tokens")
            .groupby(["user.id", "tweet.tokens"])
            .size()
            .rename("frequency")
            .reset_index()
            .rename(columns={"tweet.tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .compute()
        )

        tweet_vocabulary.to_json(
            target_path / "tweet_vocabulary.json.gz",
            compression="gzip",
            orient="records",
            lines=True,
        )

    if not (target_path / "tweet_token_frequency.json.gz").exists():
        if tweet_vocabulary is None:
            tweet_vocabulary = pd.read_json(
                target_path / "tweet_vocabulary.json.gz", lines=True
            )

        (
            tweet_vocabulary.groupby("token")
            .agg(total_frequency=("frequency", "sum"), total_users=("user.id", "count"))
            .reset_index()
            .to_json(
                target_path / "tweet_token_frequency.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

        tweet_vocabulary = None

    if not (target_path / "retweet_counts.json.gz").exists():
        (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["rt.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "retweet_counts.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "quote_counts.json.gz").exists():
        (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["quote.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "quote_counts.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "reply_counts.json.gz").exists():
        (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["in_reply_to_status_id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "reply_counts.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "retweet_edgelist.json.gz").exists():
        (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "retweet_edgelist.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "quote_edgelist.json.gz").exists():
        (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "quote_edgelist.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "reply_edgelist.json.gz").exists():
        (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "reply_edgelist.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "user_urls.json.gz").exists():
        (
            tweets[tweets["entities.urls"].notnull()]
            .drop_duplicates("id")[["user.id", "entities.urls"]]
            .assign(**{"entities.urls": lambda x: x["entities.urls"].str.split("|")})
            .explode("entities.urls")
            .assign(domain=lambda x: x["entities.urls"].map(get_domain))
            .pipe(lambda x: x[~x["domain"].isin(DISCARD_URLS)])
            .groupby(["user.id", "domain"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
            .to_json(
                target_path / "user_urls.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )

    if not (target_path / "daily_stats.json.gz").exists():
        all_tweets = (
            tweets[
                ["id", "user.id", "rt.user.id", "quote.user.id", "in_reply_to_user_id"]
            ]
            .drop_duplicates(subset="id")
            .compute()
        )

        user_stats = (
            all_tweets.set_index("user.id")
            .astype(bool)
            .reset_index()
            .groupby("user.id")
            .sum()
        )

        popularity = (
            all_tweets[all_tweets["rt.user.id"] > 0]
            .groupby("rt.user.id")
            .size()
            .rename("data.rts_received")
        )

        quotability = (
            all_tweets[all_tweets["quote.user.id"] > 0]
            .groupby("quote.user.id")
            .size()
            .rename("data.quotes_received")
        )

        conversation = (
            all_tweets[all_tweets["in_reply_to_user_id"] > 0]
            .groupby("in_reply_to_user_id")
            .size()
            .rename("data.replies_received")
        )

        user_stats = (
            user_stats.join(popularity, how="left")
            .join(quotability, how="left")
            .join(conversation, how="left")
            .fillna(0)
            .astype(int)
            .rename(
                columns={
                    "id": "data.statuses_count",
                    "rt.user.id": "data.rts_count",
                    "quote.user.id": "data.quotes_count",
                    "in_reply_to_user_id": "data.replies_count",
                }
            )
        )

        (
            pd.read_json(target_path / "unique_users.json.gz", lines=True)[
                [
                    "user.id",
                    "user.followers_count",
                    "user.friends_count",
                    "user.statuses_count",
                ]
            ]
            .set_index("user.id")
            .join(user_stats, how="inner")
            .reset_index()
            .to_json(
                target_path / "user_daily_stats.json.gz",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()