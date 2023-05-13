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

    source_path = Path(config["path"]["data"]) / "raw" / "parquet"
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
        Path(config["path"]["data"])
        / "processed"
        / "parquet"
        / experimental_settings.get("key")
    )

    # user_data = pd.read_json(
    #     processed_path / "consolidated/user.consolidated_groups.parquet", lines=True
    # )
    # user_daily_stats = pd.read_json(
    #     processed_path / "consolidated/user.daily_stats.parquet", lines=True
    # )

    user_data = dd.read_parquet(
        processed_path / "consolidated/user.consolidated_groups.parquet",
        engine="pyarrow",
    )
    user_daily_stats = dd.read_parquet(
        processed_path / "consolidated/user.daily_stats.parquet",
        engine="pyarrow",
    )

    user_content_volume = user_daily_stats.groupby("user.id")[
        [
            "data.statuses_count",
            "data.rts_count",
            "data.quotes_count",
            "data.replies_count"
            # "data.rts_received",
            # "data.quotes_received",
            # "data.replies_received",
        ]
    ].sum()

    user_transformed_volume = np.log(user_content_volume + 1).add_prefix(
        "feature.transformed_"
    )

    user_data["feature.ratio_friends_over_followers"] = np.log(
        user_data["user.friends_count"] + 1
    ) / np.log(user_data["user.followers_count"] + 1)

    user_data["feature.n_digits_username"] = (
        user_data["user.screen_name"]
        .astype(str)
        .fillna("")
        .apply(lambda x: sum(c.isdigit() for c in x), meta=("user.screen_name", "str"))
    )

    # user_data["feature.default_profile_image"] = (
    #     user_data["user.default_profile_image"].astype(int).fillna(0)
    # )

    print(user_data.columns)

    # TO DO - ERROR:
    # raise TypeError(f"Invalid value '{str(value)}' for dtype {self.dtype}")
    # TypeError: Invalid value '' for dtype Float64
    for col in user_data.columns:
        if user_data[col].dtype == "Float64":
            user_data[col].fillna(0)
        elif user_data[col].dtype == object:
            user_data[col].fillna("")

    user_features = user_data.fillna("").join(user_transformed_volume, on="user.id")

    for network in ["retweet", "quote", "reply"]:
        network_components = dd.read_parquet(
            processed_path
            / f"consolidated/network.{network}_filtered_node_components.parquet"
        ).set_index("index")

        component_count = network_components["network_component"].value_counts()
        component_count = component_count[component_count >= 10]

        print(component_count)

        network_components = (
            network_components[
                network_components["network_component"].isin(component_count.index)
            ]
            .rename({"network_component": f"feature.{network}_component"}, axis=1)
            .pipe(
                lambda x: pd.get_dummies(
                    x[f"feature.{network}_component"],
                    prefix=f"feature.{network}_component",
                )
            )
        )

        print(network_components.head())

        user_features = user_features.join(
            network_components,
            on="user.id",
            how="left",
        ).fillna(0)

    user_features = user_features.join(
        user_daily_stats.groupby("user.id").size().rename("feature.active_days"),
        on="user.id",
        how="left",
    )

    user_features["feature.daily_rythm"] = (
        user_features["user.dataset_tweets"] / user_features["feature.active_days"]
    )

    if "account_age_reference" in experimental_settings["anomalies"]:
        user_features["__ref_age__"] = datetime.datetime(
            *experimental_settings["anomalies"]["account_age_reference"]
        )
        acc_age = pd.to_datetime(user_features["__ref_age__"]) - pd.to_datetime(
            user_features["user.created_at"]
        )
        user_features["feature.account_age_days"] = acc_age.dt.days
        del user_features["__ref_age__"]

        user_features["feature.global_daily_rythm"] = np.log(
            user_features["user.statuses_count"] + 1
        ) / (user_features["feature.account_age_days"] + 1)

        user_features["feature.global_follower_rythm"] = np.log(
            user_features["user.followers_count"] + 1
        ) / (user_features["feature.account_age_days"] + 1)

        user_features["feature.global_friend_rythm"] = np.log(
            user_features["user.friends_count"] + 1
        ) / (user_features["feature.account_age_days"] + 1)

        # make these values relative
        for col in [
            "feature.transformed_data.statuses_count",
            "feature.transformed_data.rts_count",
            "feature.transformed_data.quotes_count",
            "feature.transformed_data.replies_count",
        ]:
            user_features[col] = user_features[col] / (
                user_features["feature.active_days"]
            )

    user_daily_max = user_daily_stats.loc[
        user_daily_stats.groupby(["user.id"])["date"].idxmax()
    ]
    user_daily_min = user_daily_stats.loc[
        user_daily_stats.groupby(["user.id"])["date"].idxmin()
    ]
    user_daily_diff = pd.concat([user_daily_max, user_daily_min])

    def shift(df, column):
        start_ = df.sort_values(["user.id", "date"])[[column, "user.id", "date"]]
        end_ = (
            df.sort_values(["user.id", "date"])
            .rename(
                columns={
                    column: column + "_end",
                    "user.id": "user.id_end",
                    "date": "date_end",
                }
            )[[column + "_end", "user.id_end", "date_end"]]
            .shift(-1)
        )
        shifted = (
            start_.join(end_)
            .dropna()
            .pipe(lambda x: x[x["user.id"] == x["user.id_end"]])
        )
        shifted[column + "_diff"] = shifted[column + "_end"] - shifted[column]

        return shifted[["user.id", column + "_diff", "date_end"]]

    user_daily_status_diff = pd.merge(
        shift(user_daily_diff, "user.statuses_count"),
        user_daily_max[["user.id", "predicted.stance"]],
        on="user.id",
        how="left",
    )
    user_daily_status_diff = pd.merge(
        user_daily_status_diff,
        user_features[["user.id"]],
        on="user.id",
        how="inner",
    )

    user_daily_followers_diff = pd.merge(
        shift(user_daily_diff, "user.followers_count"),
        user_daily_max[["user.id", "predicted.stance"]],
        on="user.id",
        how="left",
    )

    user_daily_followers_diff = pd.merge(
        user_daily_followers_diff,
        user_features[["user.id"]],
        on="user.id",
        how="inner",
    )

    user_daily_friends_diff = pd.merge(
        shift(user_daily_diff, "user.friends_count"),
        user_daily_max[["user.id", "predicted.stance"]],
        on="user.id",
        how="left",
    )
    user_daily_friends_diff = pd.merge(
        user_daily_friends_diff,
        user_features[["user.id"]],
        on="user.id",
        how="inner",
    )

    feature_matrix = (
        user_features.filter(like="feature.", axis=1)
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .assign(**{"user.id": user_features["user.id"]})
    )

    feature_matrix = feature_matrix.set_index("user.id")
    feature_matrix = feature_matrix.fillna(0)

    print(experimental_settings["anomalies"])

    model = IsolationForest(
        n_estimators=experimental_settings["anomalies"].get("n_estimators", 100),
        max_samples=experimental_settings["anomalies"].get("max_samples", 1000),
        contamination=experimental_settings["anomalies"].get("contamination", "auto"),
        max_features=experimental_settings["anomalies"].get("max_features", 1.0),
        n_jobs=int(experimental_settings["anomalies"].get("n_jobs", 2)),
        random_state=experimental_settings["anomalies"].get("random_state", 666),
        verbose=experimental_settings["anomalies"].get("verbose", 1),
    )
    model.fit(feature_matrix.values)

    feature_matrix["anomaly.score"] = model.decision_function(feature_matrix.values)

    results_matrix = feature_matrix.join(
        user_data[["user.id", "predicted.stance", "user.screen_name"]].set_index(
            "user.id"
        )
    )

    results_matrix["anomaly.label"] = results_matrix["anomaly.score"].apply(
        lambda x: "anomalous" if x < 0 else "normal"
    )
    print(results_matrix["anomaly.label"].value_counts(normalize=True))

    results_matrix.to_csv(
        processed_path / "consolidated" / "user.anomaly_features.csv.gz",
        compression="gzip",
    )

    joblib.dump(model, processed_path / "consolidated" / "user.anomaly_model.joblib.gz")
    print("model saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
