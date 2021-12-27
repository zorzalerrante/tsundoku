# -*- coding: utf-8 -*-
import logging
import os
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

from tsundoku.helpers import read_toml
from tsundoku.models.pipeline import classifier_pipeline, save_classifier
from gensim.utils import deaccent


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="relevance")
def main(experiment, group):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    experiment_name = experiment
    group_key = group

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

    data_base = Path(config["path"]["data"]) / "interim"
    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    with open(Path(config["path"]["config"]) / "groups" / f"{group_key}.toml") as f:
        group_config = toml.load(f)

    user_ids = pd.read_json(
        processed_path / "user.elem_ids.json.gz", lines=True
    ).set_index("user.id")
    logging.info(f"Total users: #{len(user_ids)}")

    relevance_path = processed_path / "relevance.classification.predictions.json.gz"

    xgb_parameters = experiment_config[group_key]["xgb"]
    pipeline_config = experiment_config[group_key]["pipeline"]

    if "allow_list" in experiment_config[group_key]:
        allow_list_ids = experiment_config[group_key]["allow_list"].get(
            "user_ids", None
        )
    else:
        allow_list_ids = None

    if group_key != "relevance" and relevance_path.exists():
        user_groups = pd.read_json(relevance_path, lines=True).set_index("user.id")
        # TODO: make undisclosed optional
        all_ids = user_groups.index
        valid_users = user_groups[
            ~(user_groups["predicted_class"].isin(["noise", "undisclosed"]))
        ].index

        # note that even if we discard undisclosed users, they may be present in the allow_list.
        # we check agains the full list of users
        if allow_list_ids:
            valid_users = set(valid_users) | (set(allow_list_ids) & set(all_ids))

        user_ids = user_ids.loc[valid_users].sort_values("row_id")
        logging.info(f"Relevant users for {group_key} prediction: #{len(user_ids)}")

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=group_config.keys(), dtype=int
    )

    if group_key == "location":
        import re

        user_data = (
            pd.read_json(processed_path / "user.unique.json.gz", lines=True)
            .set_index("user.id")
            .loc[user_ids.index]
        )
        user_data["user.location"] = user_data["user.location"].map(deaccent)

        for group, meta in group_config.items():
            try:
                group_re = re.compile(
                    "|".join(meta["location"]["patterns"]), re.IGNORECASE
                )
            except KeyError:
                print(f"no location patterns in {group}")
                continue

            group_ids = user_data[
                user_data["user.location"].str.contains(group_re)
            ].index
            labels[group].loc[group_ids] = 1

        user_data = None

        print(labels.sum())

    clf, predictions, feature_names_all, top_terms, X = classifier_pipeline(
        processed_path,
        group_config,
        user_ids,
        labels,
        xgb_parameters,
        allowed_user_ids=allow_list_ids,
        early_stopping_rounds=pipeline_config["early_stopping_rounds"],
        eval_fraction=pipeline_config["eval_fraction"],
        threshold_offset_factor=pipeline_config["threshold_offset_factor"],
    )
    save_classifier(
        group_key, processed_path, X, clf, predictions, feature_names_all, top_terms
    )

    for loc in top_terms.columns:
        print(loc)
        # print(top_terms.loc[relevant_features['label']].sort_values(loc, ascending=False)[loc].head(15))
        print(
            top_terms[top_terms[loc] > 10]
            .sort_values(loc, ascending=False)[loc]
            .sample(min(25, len(top_terms[top_terms[loc] > 10])))
        )
        print(
            top_terms[top_terms.index.str.contains("tweet")]
            .sort_values(loc, ascending=False)[loc]
            .head(15)
        )
        print(
            top_terms[top_terms.index.str.contains("#")]
            .sort_values(loc, ascending=False)[loc]
            .head(15)
        )
        print(
            top_terms[top_terms.index.str.contains("@")]
            .sort_values(loc, ascending=False)[loc]
            .head(15)
        )
        print(
            top_terms[top_terms.index.str.contains("domain")]
            .sort_values(loc, ascending=False)[loc]
            .head(15)
        )
        print("")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
