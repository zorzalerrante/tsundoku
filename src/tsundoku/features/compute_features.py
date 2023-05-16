# -*- coding: utf-8 -*-
import logging
import os
import click
import dask
import dask.dataframe as dd
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from tsundoku.utils.tweets import TWEET_DTYPES
from tsundoku.utils.urls import DISCARD_URLS, get_domain
from tsundoku.utils.files import read_toml, write_json
from tsundoku.utils.timer import Timer


@click.command()
@click.option("--start_at", type=str, default="")
@click.option("--overwrite", type=bool, default=False)
def main(start_at, overwrite):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")[
        "project"
    ]
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config["environment"].get("n_jobs", 2))))

    source_path = Path(config["path"]["data"]) / "raw" / "parquet"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_folders = sorted(glob(str(source_path / "*")))
    logging.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    t = Timer()
    chronometer = []
    dates = []
    for tweet_path in source_folders:
        t.start()
        date = str(os.path.basename(tweet_path))

        if start_at and date < start_at:
            continue

        target = Path(config["path"]["data"]) / "interim" / "parquet" / f"{date}"

        if not target.exists():
            target.mkdir(parents=True)
            logging.info(f"created: {tweet_path} -> {target}")
        else:
            logging.info(f"{target} already exists.")
            if not overwrite:
                logging.info(f"skipping.")
                continue

        target_files = glob(str(Path(tweet_path) / "*.parquet"))

        non_empty_files = list(filter(lambda x: os.stat(x).st_size > 10, target_files))

        if not non_empty_files:
            logger.warning(f"{date} has no validfiles.")
            continue

        tweets = dd.read_parquet(non_empty_files, schema=TWEET_DTYPES)

        if tweets.npartitions <= 0:
            logger.warning(f"{date} has no files")
            continue

        logger.info(
            f"{date} ({tweets.npartitions} partitions) -> computing user metrics with parquet files"
        )
        compute_user_metrics(tweets, target, overwrite)
        logger.info(
            f"{date} ({tweets.npartitions} partitions) -> computing tweet metrics with parquet files"
        )
        compute_tweet_metrics(tweets, target, overwrite)
        logger.info(f"{date} -> done! :D")

        current_timer = t.stop()
        chronometer.append(current_timer)
        dates.append(date)
        print(f"Succesfully computed {date} data features in {current_timer} seconds!")

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer dates: " + str(dates))


def compute_user_metrics(tweets, target_path, overwrite):
    users = None
    if overwrite or not (target_path / "unique_users.parquet").exists():
        users = (
            tweets.drop_duplicates(subset="user.id")
            .compute()
            .filter(regex="^user\.?.*")
        )

        users_table = pa.Table.from_pandas(users)
        pq.write_table(
            users_table, target_path / "unique_users.parquet", use_dictionary=False
        )

    if overwrite or not (target_path / "user_name_vocabulary.parquet").exists():
        if users is None:
            users = dd.read_parquet(target_path / "unique_users.parquet")

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
        )
        user_name_vocabulary_table = pa.Table.from_pandas(users)
        pq.write_table(
            user_name_vocabulary_table,
            target_path / "user_name_vocabulary.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "user_description_vocabulary.parquet").exists():
        if users is None:
            users = dd.read_parquet(target_path / "unique_users.parquet")

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
        )
        user_description_vocabulary_table = pa.Table.from_pandas(users)
        pq.write_table(
            user_description_vocabulary_table,
            target_path / "user_description_vocabulary.parquet",
            use_dictionary=False,
        )


def compute_tweet_metrics(tweets, target_path, overwrite):
    if overwrite or not (target_path / "tweets_per_user.parquet").exists():
        tweets_per_user = (
            tweets.drop_duplicates("id")
            .groupby("user.id")
            .size()
            .compute()
            .reset_index()
        )

        tweets_per_user.columns = tweets_per_user.columns.astype(str)
        tweets_per_user_table = pa.Table.from_pandas(tweets_per_user)
        pq.write_table(
            tweets_per_user_table,
            target_path / "tweets_per_user.parquet",
            use_dictionary=False,
        )

    tweet_vocabulary = None

    if overwrite or not (target_path / "tweet_vocabulary.parquet").exists():
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

        tweet_vocabulary_table = pa.Table.from_pandas(tweet_vocabulary)
        pq.write_table(
            tweet_vocabulary_table,
            target_path / "tweet_vocabulary.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "tweet_token_frequency.parquet").exists():
        if tweet_vocabulary is None:
            tweet_vocabulary = dd.read_parquet(target_path / "tweet_vocabulary.parquet")

        (
            tweet_vocabulary.groupby("token")
            .agg(total_frequency=("frequency", "sum"), total_users=("user.id", "count"))
            .reset_index()
            .to_json(
                target_path / "tweet_token_frequency.parquet",
                compression="gzip",
                orient="records",
                lines=True,
            )
        )
        tweet_token_frequency_table = pa.Table.from_pandas(tweet_vocabulary)
        pq.write_table(
            tweet_token_frequency_table,
            target_path / "tweet_token_frequency.parquet",
            use_dictionary=False,
        )

        tweet_vocabulary = None

    if overwrite or not (target_path / "retweet_counts.parquet").exists():
        retweet_counts = (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["rt.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        retweet_counts_table = pa.Table.from_pandas(retweet_counts)
        pq.write_table(
            retweet_counts_table,
            target_path / "retweet_counts.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "quote_counts.parquet").exists():
        quote_counts = (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["quote.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        quote_counts_table = pa.Table.from_pandas(quote_counts)
        pq.write_table(
            quote_counts_table,
            target_path / "quote_counts.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "reply_counts.parquet").exists():
        reply_counts = (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["in_reply_to_status_id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        reply_counts_table = pa.Table.from_pandas(reply_counts)
        pq.write_table(
            reply_counts_table,
            target_path / "reply_counts.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "retweet_edgelist.parquet").exists():
        retweet_edgelist = (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        retweet_edgelist_table = pa.Table.from_pandas(retweet_edgelist)
        pq.write_table(
            retweet_edgelist_table,
            target_path / "retweet_edgelist.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "quote_edgelist.parquet").exists():
        quote_edgelist = (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        quote_edgelist_table = pa.Table.from_pandas(quote_edgelist)
        pq.write_table(
            quote_edgelist_table,
            target_path / "quote_edgelist.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "reply_edgelist.parquet").exists():
        reply_edgelist = (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )
        reply_edgelist_table = pa.Table.from_pandas(reply_edgelist)
        pq.write_table(
            reply_edgelist_table,
            target_path / "reply_edgelist.parquet",
            use_dictionary=False,
        )

    if overwrite or not (target_path / "user_urls.parquet").exists():
        user_urls = (
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
        )
        user_urls_table = pa.Table.from_pandas(user_urls)
        pq.write_table(
            user_urls_table, target_path / "user_urls.parquet", use_dictionary=False
        )

    if overwrite or not (target_path / "daily_stats.parquet").exists():
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

        plain = (
            all_tweets[
                (all_tweets["rt.user.id"] == 0)
                & (all_tweets["quote.user.id"] == 0)
                & (all_tweets["in_reply_to_user_id"] == 0)
            ]
            .groupby("user.id")
            .size()
            .rename("data.plain_count")
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
            .join(plain, how="left")
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

        user_daily_stats = (
            dd.read_parquet(target_path / "unique_users.parquet")[
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
        )
        user_daily_stats.to_parquet(
            target_path,
            name_function=lambda i: f"user_daily_stats{f'_{i}' if i != 0 else ''}.parquet",
            engine="pyarrow",
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
