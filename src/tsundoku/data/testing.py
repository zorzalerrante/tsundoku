# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import dask
import dask.dataframe as dd

from dotenv import find_dotenv, load_dotenv

from tsundoku.data.importer import TweetImporter
import gzip


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    project = TweetImporter(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")
    logger.info(str(project.config))

    target_path = project.data_path() / "raw" / "parquet" / "2022-01-02" / \
        "tweets.partition.0.parquet"

    ddd = dd.read_parquet(target_path)
    print(ddd.head())
    return ddd


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
