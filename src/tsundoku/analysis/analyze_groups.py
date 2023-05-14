import logging
import os
import click
import dask
import dask.dataframe as dd
import graph_tool
import graph_tool.topology
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import toml

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from aves.models.network import Network

# from scipy.sparse import dok_matrix, save_npz

# from tsundoku.utils.vocabulary import build_elem_to_id, filter_vocabulary
# from tsundoku.utils.dtm import build_vocabulary, tokens_to_document_term_matrix
# from tsundoku.utils.tweets import TWEET_DTYPES
# from tsundoku.utils.urls import DISCARD_URLS, get_domain
from tsundoku.utils.users import USERS_DTYPES
from tsundoku.utils.files import read_toml, write_json


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="relevance")
@click.option("--overwrite", type=bool, default=False)
def main(experiment, group, overwrite):
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

    data_base = Path(config["path"]["data"]) / "interim" / "parquet"
    data_paths = [data_base / key for key in key_folders]
    processed_path = (
        Path(config["path"]["data"])
        / "processed"
        / "parquet"
        / experimental_settings.get("key")
    )
    target_path = processed_path / "consolidated"

    if not target_path.exists():
        target_path.mkdir(parents=True)
        logging.info(f"{target_path} created")

    user_ids = consolidate_users(
        processed_path, target_path, group, overwrite=overwrite
    )
    aggregate_daily_stats(
        data_paths, processed_path, user_ids, target_path, group, overwrite=overwrite
    )
    sum_word_frequencies_per_group(
        data_paths, processed_path, target_path, group, overwrite=overwrite
    )

    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "rt.user.id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="retweet",
    )

    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "quote.user.id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="quote",
    )

    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "in_reply_to_user_id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="reply",
    )


def read_daily_stats(source_folder, user_ids):
    date = os.path.basename(source_folder)

    return (
        dd.read_parquet(source_folder / "user_daily_stats.parquet")
        .pipe(lambda x: x[x["user.id"].isin(user_ids)])
        .assign(date=date)
    )


def aggregate_daily_stats(
    data_paths,
    processed_path,
    user_ids,
    target_path,
    aggregation_group,
    overwrite=False,
):
    daily_stats_target_path = target_path / "user.daily_stats.parquet"

    if not overwrite and daily_stats_target_path.exists():
        logging.info(
            f"Daily Stats already exists. Delete if you want to build it again -> {daily_stats_target_path}"
        )
        return

    tasks = []

    for folder in data_paths:
        if (folder / "user_daily_stats.parquet").exists():
            tasks.append(dask.delayed(read_daily_stats)(folder, user_ids).compute())
        else:
            logging.info(f"Daily Stats: {folder} does not exist")

    results = pd.concat(dask.compute(tasks)[0]).sort_values("date").set_index("user.id")

    user_groups = (
        dd.read_parquet(
            processed_path / f"{aggregation_group}.classification.predictions.parquet"
        )
        .pipe(lambda x: x[x["predicted_class"] != "noise"])
        .set_index("user.id")["predicted_class"]
        .rename(f"predicted.{aggregation_group}")
        .compute()
    )

    (
        results.join(user_groups, how="inner").to_parquet(
            daily_stats_target_path,
            engine="pyarrow",
            use_dictionary=False,
        )
    )

    logging.info(f"Daily Stats -> {daily_stats_target_path}")


def count_group_vocabulary(source_folder, user_stances, aggregation_group):
    date = os.path.basename(source_folder)

    user_vocab = (
        dd.read_parquet(source_folder / "tweet_vocabulary.parquet")
        .join(user_stances, on="user.id")
        .groupby([f"predicted.{aggregation_group}", "token"], sort=False)
        .agg(n_users=("user.id", "count"), frequency=("frequency", "sum"))
        .assign(date=date)
        .reset_index()
    )

    return user_vocab


def sum_word_frequencies_per_group(
    data_paths, processed_path, target_path, aggregation_group, overwrite=False
):
    frequencies_target_path = target_path / "tweet.word_frequencies.parquet"

    if not overwrite and frequencies_target_path.exists():
        logging.info(
            f"Daily Frequencies already exists. Delete if you want to build it again -> {frequencies_target_path}"
        )
        return

    user_groups = (
        dd.read_parquet(
            processed_path / f"{aggregation_group}.classification.predictions.parquet"
        )
        .pipe(lambda x: x[x["predicted_class"] != "noise"])
        .set_index("user.id")["predicted_class"]
        .rename(f"predicted.{aggregation_group}")
    )

    tasks = []

    for folder in data_paths:
        if (folder / "tweet_vocabulary.parquet").exists():
            tasks.append(
                dask.delayed(count_group_vocabulary)(
                    folder, user_groups, aggregation_group
                ).compute()
            )
        else:
            logging.info(f"Daily Frequencies: {folder} does not exist")

    results = pd.concat(dask.compute(tasks)[0]).sort_values("date")

    (
        results.to_parquet(
            frequencies_target_path,
            engine="pyarrow",
            use_dictionary=False,
        )
    )
    logging.info(f"Daily Frequencies -> {frequencies_target_path}")


def consolidate_users(processed_path, target_path, aggregation_group, overwrite=False):
    users_target_path = target_path / "user.consolidated_groups.parquet"

    if not overwrite and users_target_path.exists():
        logging.info(
            f"Consolidated users file already exists. Delete if you want to build it again -> {users_target_path}"
        )
        return dd.read_parquet(users_target_path, lines=True)["user.id"].values

    user_groups = dd.read_parquet(
        processed_path / f"{aggregation_group}.classification.predictions.parquet"
    ).set_index("user.id")

    invalid_users = set(user_groups[user_groups["predicted_class"] == "noise"].index)

    users = (
        dd.read_parquet(processed_path / "user.unique.parquet", schema=USERS_DTYPES)
        .reset_index()
        .rename(columns={"index": "row_id"})
        .pipe(lambda x: x[~x["user.id"].isin(invalid_users)])
        .set_index("user.id")
        .join(
            user_groups["predicted_class"].rename(f"predicted.{aggregation_group}"),
            how="inner",
        )
    )

    for group_name in ["location", "person"]:
        predictions_path = (
            processed_path / f"{group_name}.classification.predictions.parquet"
        )

        if not predictions_path.exists():
            logging.info(
                f"prediction type {group_name} not found (no file: {predictions_path}"
            )
            continue

        user_predictions = dd.read_parquet(predictions_path).set_index("user.id")
        invalid_users = invalid_users | set(
            user_predictions[user_predictions["predicted_class"] == "noise"].index
        )
        users = users.join(
            user_predictions["predicted_class"].rename(f"predicted.{group_name}"),
            how="inner",
        )
        logging.info(
            f"prediction type {group_name} found! updated #users: {len(users)}"
        )

    users = users.reset_index().pipe(lambda x: x[~x["user.id"].isin(invalid_users)])

    logging.info(f"users: #valid: {len(users)}, #invalid {len(invalid_users)}")
    logging.info(f"user consolidation with groups -> {users_target_path}")

    users.to_parquet(
        target_path,
        name_function=lambda i: f"user.consolidated_groups{f'_{i}' if i != 0 else ''}.parquet",
        engine="pyarrow",
        schema=USERS_DTYPES,
        use_dictionary=False,
    )

    return users.compute()["user.id"].values


def identify_network_lcc(
    processed_path,
    target_path,
    user_ids,
    target_column,
    min_freq=1,
    network_type="retweet",
):
    edges_file = processed_path / f"user.{network_type}_edges.all.parquet"

    edges = (
        dd.read_parquet(edges_file, engine="pyarrow")
        .pipe(lambda x: x[x["user.id"].isin(user_ids)])
        .pipe(lambda x: x[x["frequency"] >= min_freq])
    )

    unique_user_ids = set(edges["user.id"].unique()) | set(
        edges[target_column].unique()
    )
    id_to_node = dict(zip(unique_user_ids, range(len(unique_user_ids))))
    logging.info(
        f"network type {network_type}: #{len(unique_user_ids)} nodes, #{len(edges)} edges"
    )

    edges["source"] = edges["user.id"].map(id_to_node)
    edges["target"] = edges[target_column].map(id_to_node)
    network = Network.from_edgelist(edges.compute(), weight="frequency")
    graph = network.network

    components, component_histogram = graph_tool.topology.label_components(
        graph, directed=False
    )
    logging.info(f"network components: {component_histogram}")

    node_components = pd.Series(
        components.a.tolist(), index=list(id_to_node.keys()), name="network_component"
    )
    node_components_path = (
        target_path / f"network.{network_type}_filtered_node_components.parquet"
    )

    node_components_table = pa.Table.from_pandas(node_components.reset_index())
    pq.write_table(
        node_components_table,
        node_components_path,
        use_dictionary=False,
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
