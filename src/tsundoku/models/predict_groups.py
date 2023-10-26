import logging
import os
import re
import sys
import click
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gensim.utils import deaccent

from tsundoku.utils.files import read_toml
from tsundoku.models.pipeline import classifier_pipeline, save_classifier
from tsundoku.utils.timer import Timer


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="relevance")
def main(experiment, group):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed/parquet)
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

    source_path = Path(config["path"]["data"]) / "raw"
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

    group_annotations_file = Path(config["path"]["config"]) / "groups" / f"{group_key}.annotations.csv"

    if group_annotations_file.exists():
        logging.info('Reading annotations...')
        group_annotations = pd.read_csv(group_annotations_file)
        
        for key in group_config.keys():
            annotated_user_ids = group_annotations[group_annotations['class'] == key]
            if not annotated_user_ids.empty:
                logging.info(f'# of annotated "{key}" accounts: {len(annotated_user_ids)}')
                group_config[key]['account_ids']['known_users'].extend(annotated_user_ids['user.id'].unique()) 

    user_ids = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")
        .compute()
    )
    logging.info(f"Total users: #{len(user_ids)}")

    relevance_path = processed_path / "relevance.classification.predictions.parquet"

    xgb_parameters = experiment_config[group_key]["xgb"]
    pipeline_config = experiment_config[group_key]["pipeline"]

    if "allow_list" in experiment_config[group_key]:
        allow_list_ids = experiment_config[group_key]["allow_list"].get(
            "user_ids", None
        )
        allow_id_class = experiment_config[group_key]["allow_list"].get(
            "assigned_class", "undisclosed"
        )
        logging.info(
            f"Whitelisted accounts: #{len(allow_list_ids)}. Using class {allow_id_class}"
        )
    else:
        allow_list_ids = None
        allow_id_class = None
        logging.info(f"No whitelisted accounts")

    if group_key != "relevance" and relevance_path.exists():
        user_groups = dd.read_parquet(relevance_path).set_index("user.id")
        # TODO: make undisclosed optional
        all_ids = user_groups.index
        valid_users = user_groups[
            ~(user_groups["predicted_class"].isin(["noise", "undisclosed"]))
        ].index

        # note that even if we discard undisclosed users, they may be present in the allow_list.
        # we check agains the full list of users
        if allow_list_ids is not None:
            valid_users = set(valid_users) | (set(allow_list_ids) & set(all_ids))

        user_ids = user_ids.loc[valid_users].sort_values("row_id")
        logging.info(f"Relevant users for {group_key} prediction: #{len(user_ids)}")

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=group_config.keys(), dtype=int
    )

    # if there are location patterns, tream them specially:
    user_data = None

    def load_user_data():
        return (
            dd.read_parquet(processed_path / "user.unique.parquet")
            .set_index("user.id")
            .loc[user_ids.index]
            .compute()
        )

    for key, meta in group_config.items():
        group_re = None
        try:
            print(f'location patterns for {key}, {meta["location"]["patterns"]}')
            group_re = re.compile("|".join(meta["location"]["patterns"]), re.IGNORECASE)
        except KeyError:
            print(f"no location patterns in {key}")
            continue

        if user_data is None:
            user_data = load_user_data()
            user_data["user.location"] = (
                user_data["user.location"].fillna("").map(deaccent)
            )

        group_ids = user_data[user_data["user.location"].str.contains(group_re)].index

        if group == "location":
            # use these as account ids that cannot be modified (let's trust users)
            if not "account_ids" in meta:
                meta["account_ids"] = dict()

            if not "known_users" in meta:
                meta["account_ids"]["known_users"] = list(group_ids)
            else:
                meta["account_ids"]["known_users"].extend(group_ids)
        else:
            # use them as labels
            labels[key].loc[group_ids] = 1

    # special case: age
    if group == "age":
        if user_data is None:
            user_data = load_user_data()

        min_age = 10
        max_age = 90

        def pick_age(values):
            if not values:
                return 0
            # print(values)
            for x in values[0]:
                if not x or int(x) < min_age:
                    continue
                return int(x)
            return 0

        age_patterns = re.compile(
            "(?:^|level|lvl|nivel)\s?([0-6][0-9])\.|(?:^|\W)([0-9]{2})\s?(?:años|veranos|otoños|inviernos|primaveras|years old|vueltas|lunas|soles)",
            flags=re.IGNORECASE | re.UNICODE,
        )

        found_age = (
            user_data["user.description"]
            .fillna("")
            .str.findall(age_patterns)
            .map(pick_age)
        )
        found_age = found_age[found_age.between(min_age, max_age)].copy()
        print("found_age", found_age.shape)

        print(found_age.sample(10))
        print(found_age.value_counts())

        labeled_age = pd.cut(
            found_age, bins=[0, 17, 29, 39, 49, max_age + 1], labels=group_config.keys()
        )

        print(labeled_age.value_counts())

        print(labeled_age.sample(10))

        for key in group_config.keys():
            print(key)
            print((labeled_age == key).index)
            labels[key].loc[labeled_age[labeled_age == key].index] = 1

        skip_numeric_tokens = True
    else:
        skip_numeric_tokens = False

    if user_data is not None:
        user_data = None

    print(labels.sample(10))
    print(labels.sum())

    t = Timer()
    chronometer = []
    process_names = []
    t.start()
    clf, predictions, feature_names_all, top_terms, X = classifier_pipeline(
        processed_path,
        group_config,
        user_ids,
        labels,
        xgb_parameters,
        allowed_user_ids=allow_list_ids,
        allowed_users_class=allow_id_class,
        early_stopping_rounds=pipeline_config["early_stopping_rounds"],
        eval_fraction=pipeline_config["eval_fraction"],
        threshold_offset_factor=pipeline_config["threshold_offset_factor"],
        skip_numeric_tokens=skip_numeric_tokens,
    )

    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"classification")

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
            .head()
        )
        print(
            top_terms[top_terms.index.str.contains("tweet")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        print(
            top_terms[top_terms.index.str.contains("#")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        print(
            top_terms[top_terms.index.str.contains("@")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        print(
            top_terms[top_terms.index.str.contains("domain")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        print("")

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer process names: " + str(process_names))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
