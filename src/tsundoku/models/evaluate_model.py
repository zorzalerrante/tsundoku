# -*- coding: utf-8 -*-
import copy
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
import toml
from dotenv import find_dotenv, load_dotenv
from scipy.sparse import dok_matrix, save_npz

from tsundoku.features.analysis import build_elem_to_id, filter_vocabulary
from tsundoku.features.dtm import build_vocabulary, tokens_to_document_term_matrix
from tsundoku.features.tweets import TWEET_DTYPES
from tsundoku.features.urls import DISCARD_URLS, get_domain
from tsundoku.helpers import read_json, write_json
from tsundoku.models.pipeline import evaluate, prepare_features


@click.command()
@click.argument("experiment_name", type=str)
@click.argument("group_key", type=str)
@click.option("--n_splits", default=5, type=int)
def main(experiment_name, group_key, n_splits):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_json(Path(os.environ["TSUNDOKU_PROJECT-PATH"]) / "config.json")
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config.get("n_jobs", 2))))

    source_path = Path(config["data_path"]) / "raw" / "json"
    experiment_file = Path(config["project_path"]) / "experiments" / "full.toml"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not experiment_file.exists():
        raise FileNotFoundError(experiment_file)

    with open(experiment_file) as f:
        experiment_config = toml.load(f)
        logging.info(f"{experiment_config}")

    experimental_settings = experiment_config["experiments"][experiment_name]
    logging.info(f"Experimental settings: {experimental_settings}")

    source_folders = sorted(
        glob(str(source_path / experimental_settings.get("folder_pattern", "*")))
    )
    logging.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    key_folders = map(os.path.basename, source_folders)

    if experimental_settings.get("folder_start", None) is not None:
        key_folders = filter(
            lambda x: x >= experimental_settings.get("folder_start"), key_folders
        )

    if experimental_settings.get("folder_end", None) is not None:
        key_folders = filter(
            lambda x: x <= experimental_settings.get("folder_end"), key_folders
        )

    key_folders = list(key_folders)
    logging.info(f"{key_folders}")

    # let's go

    data_base = Path(config["data_path"]) / "interim"
    processed_path = (
        Path(config["data_path"]) / "processed" / experimental_settings.get("key")
    )

    with open(Path(config["project_path"]) / "groups" / f"{group_key}.toml") as f:
        group_config = toml.load(f)

    # these are sorted by tweet count!
    user_ids = pd.read_json(
        processed_path / "user.elem_ids.json.gz", lines=True
    ).set_index("user.id")
    logging.info(f"Total users: #{len(user_ids)}")

    # user_ids['rank_quartile'] = pd.qcut(user_ids['row_id'], 20, retbins=False, labels=range(20))

    # we discard noise to evaluate, including in stance classification!
    user_groups = pd.read_json(
        processed_path / "stance.classification.predictions.json.gz", lines=True
    ).set_index("user.id")
    valid_users = user_groups[user_groups["predicted_class"] != "noise"].index
    user_ids = user_ids.loc[valid_users].sort_values("row_id")
    logging.info(f"Kept users for {group_key} prediction: #{len(user_ids)}")

    columns = [g for g in group_config.keys() if g != "noise"]

    # if 'noise' in group_config:
    #    del group_config['noise']

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=group_config.keys(), dtype=int
    )

    xgb_parameters = experiment_config[group_key]["xgb"]
    pipeline_config = experiment_config[group_key]["pipeline"]

    X, labels, feature_names_all = prepare_features(
        processed_path, group_config, user_ids, labels
    )

    outputs = evaluate(
        processed_path,
        xgb_parameters,
        X,
        labels,
        group_key,
        training_eval_fraction=pipeline_config["eval_fraction"],
        n_splits=n_splits,
    )
    logging.info(f"{str(outputs)}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
