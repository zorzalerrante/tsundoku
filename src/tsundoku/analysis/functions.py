import logging
import os
import dask
import dask.dataframe as dd
import graph_tool
import graph_tool.topology
import pandas as pd
import pyarrow as pa

from aves.models.network import Network

from tsundoku.utils.users import USERS_DTYPES
from tsundoku.utils.files import write_parquet


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


def consolidate_users(processed_path, target_path, aggregation_group, overwrite=False, group_names=None):
    if group_names is None:
        group_names = []

    users_target_path = target_path / "user.consolidated_groups.parquet"

    if not overwrite and users_target_path.exists():
        logging.info(
            f"Consolidated users file already exists. Delete if you want to build it again -> {users_target_path}"
        )
        return dd.read_parquet(users_target_path, lines=True)["user.id"].values

    user_groups = (
        dd.read_parquet(
            processed_path / f"{aggregation_group}.classification.predictions.parquet"
        )
        .set_index("user.id")
        .compute()
    )

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

    schema = USERS_DTYPES.append(
        pa.field(f"predicted.{aggregation_group}", pa.string())
    )

    user_predictions = None

    for group_name in group_names:
        if group_name == 'relevance' or group_name == aggregation_group:
            continue

        predictions_path = (
            processed_path / f"{group_name}.classification.predictions.parquet"
        )

        if not predictions_path.exists():
            logging.info(
                f"prediction type {group_name} not found (no file: {predictions_path}"
            )
            continue

        schema = schema.append(pa.field(f"predicted.{group_name}", pa.string()))

        group_predictions = (
            dd.read_parquet(predictions_path)
            .set_index("user.id")
            .rename(columns={"predicted_class": f"predicted.{group_name}"})
            .compute()
        )

        invalid_users = invalid_users | set(
            group_predictions[
                group_predictions[f"predicted.{group_name}"] == "noise"
            ].index
        )

        group_predictions = group_predictions.pipe(
            lambda x: x[x[f"predicted.{group_name}"] != "noise"]
        )

        if user_predictions is None:
            user_predictions = group_predictions[[f"predicted.{group_name}"]]
        else:
            user_predictions = user_predictions.join(
                group_predictions[[f"predicted.{group_name}"]], how="inner"
            )

        logging.info(f"prediction type {group_name} found!")

    users = users.join(user_predictions, how="inner").reset_index()

    logging.info(f"users: #valid: {len(users)}, #invalid {len(invalid_users)}")
    logging.info(f"user consolidation with groups -> {users_target_path}")

    users.to_parquet(
        target_path,
        name_function=lambda i: f"user.consolidated_groups{f'_{i}' if i != 0 else ''}.parquet",
        engine="pyarrow",
        schema=schema,
        use_dictionary=False,
    )

    return users["user.id"].compute().values


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

    write_parquet(node_components.reset_index(), node_components_path)
