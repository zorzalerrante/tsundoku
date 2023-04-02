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

import ctypes
from tsundoku.features.timer import Timer


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info("Testing")

    t = Timer()
    t.start()
    list = [1, 2, 3]
    list2 = [1, 2, 3]

    logger.info("Testing: " + str(list))
    logger.info("Testing: " + str(list2))

    time = t.stop()

    logger.info("time: " + str(time))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
