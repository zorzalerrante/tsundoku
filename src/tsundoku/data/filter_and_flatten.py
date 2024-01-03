# -*- coding: utf-8 -*-
import click
import logging
import glob
import rapidjson as json
import os
import os.path
import gzip
import zlib
import dask

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from multiprocessing.pool import ThreadPool

dask.config.set(pool=ThreadPool(2))
from tsundoku.utils.iterator import iterate_tweets


@click.command()
def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("flattening incoming data")

    files = sorted(glob.glob(f"{os.environ['INCOMING_PATH']}/*.gz"))
    target = Path(os.environ["JSON_TWEET_PATH"])

    languages = list(os.environ.get("TSUNDOKU_LANGUAGES", "es|und").split("|"))
    logger.info(f"accepted languages: {languages}")
    languages.append(None)

    if not target.exists():
        target.mkdir(parents=True)
        logger.info(f"{target} created")

    def filter_tweets(filename):
        source_file = filename
        target_file = target / os.path.basename(filename)
        prev_size = os.stat(source_file).st_size / 1024 / 1024
        # print(source_file, )
        # print(target_file)

        written_tweets = 0
        total_tweets = 0

        try:
            with gzip.open(target_file, "wt") as dst:
                with gzip.open(source_file, "r") as src:
                    for tweet in iterate_tweets(src):
                        total_tweets += 1
                        if not tweet.get("lang", None) in languages:
                            continue

                        # print(tweet["lang"], tweet["text"])
                        # break
                        dst.write(f"{json.dumps(tweet, ensure_ascii=False)}\n")
                        written_tweets += 1

            full_size = os.stat(target_file).st_size / 1024 / 1024
            logger.info(
                f"{target_file} ({full_size:.2f} of {prev_size:.2f} MBs, {written_tweets}/{total_tweets} tweets)"
            )
            os.unlink(source_file)
        except zlib.error:
            logger.error(
                f"{source_file} is corrupted ({written_tweets}/{total_tweets} tweets written)"
            )
            os.unlink(source_file)
        except gzip.BadGzipFile:
            logger.error(f"{source_file} is fully corrupted!")
            os.unlink(source_file)

    tasks = [dask.delayed(filter_tweets)(filename) for filename in files]
    dask.compute(*tasks)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
