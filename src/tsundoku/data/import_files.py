# -*- coding: utf-8 -*-
import logging
import os
import click

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from tsundoku.data.importer import TweetImporter


@click.command()
@click.argument("source", type=click.Path(exists=True), nargs=-1)
@click.option("--target", type=str)
@click.option("--encoding", default="utf-8", type=str)
def main(source, target, encoding):
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    project = TweetImporter(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")
    logger.info(str(project.config))

    target_path = project.data_path() / "raw" / target
    project.import_files(source, target_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
