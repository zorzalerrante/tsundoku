# -*- coding: utf-8 -*-
import datetime
import logging
import os
import re
from multiprocessing.pool import ThreadPool
from pathlib import Path

import click
import dask
import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import toml
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import IsolationForest

from tsundoku.helpers import read_toml
from aves.models.network import Network


@click.command()
@click.option("--experiment", type=str, default="full")
def main(experiment):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    experiment_name = experiment

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")[
        "project"
    ]
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config.get("n_jobs", 2))))

    source_path = Path(config["path"]["data"]) / "raw" / "json"
    experiment_file = Path(config["path"]["config"]) / "experiments.toml"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not experiment_file.exists():
        raise FileNotFoundError(experiment_file)

    with open(experiment_file) as f:
        experiment_config = toml.load(f)
        logging.info(f"{experiment_config}")

    experimental_settings = experiment_config["experiments"][experiment_name]
    logging.info(f"Experimental settings: {experimental_settings}")

    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    users = pd.read_json(
        processed_path / "consolidated" / "user.consolidated_groups.parquet", lines=True
    )

    rts = pd.read_json(
        processed_path / "user.retweet_edges.all.parquet", lines=True
    ).pipe(
        lambda x: x[
            x["user.id"].isin(users["user.id"]) & x["rt.user.id"].isin(users["user.id"])
        ]
    )

    high_participation = users[users["user.dataset_tweets"] >= 1]

    edge_list = rts[
        rts["user.id"].isin(high_participation["user.id"])
        & rts["rt.user.id"].isin(high_participation["user.id"])
    ]

    rt_network = Network.from_edgelist(
        edge_list[edge_list["frequency"] >= 1],
        source="user.id",
        target="rt.user.id",
        weight="frequency",
    )
    rt_network.network

    connected_rt_network = rt_network.largest_connected_component(directed=True)

    print(connected_rt_network.num_vertices, connected_rt_network.num_edges)

    connected_rt_network.save(
        processed_path / "consolidated" / "rt_connected.network.gt"
    )

    print('saved')

    connected_rt_network.detect_communities(
        method='hierarchical', hierarchical_covariate_type="discrete-poisson"
    )

    connected_rt_network.save(
        processed_path / "consolidated" / "communities_rt_connected.network.gt"
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
