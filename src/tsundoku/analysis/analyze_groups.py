import logging
import os
import click
import dask
import toml

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from tsundoku.utils.files import read_toml
from tsundoku.utils.timer import Timer

from .functions import (
    consolidate_users,
    aggregate_daily_stats,
    sum_word_frequencies_per_group,
    identify_network_lcc,
)


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

    group_names = glob(str(Path(config["path"]["config"]) / "groups" / "*.toml"))
    group_names = list(map(lambda x: os.path.basename(x).split('.')[0], group_names))
    logging.info(f'Group names: {group_names}')

    # let's go

    data_base = Path(config["path"]["data"]) / "interim"
    data_paths = [data_base / key for key in key_folders]
    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )
    target_path = processed_path / "consolidated"

    if not target_path.exists():
        target_path.mkdir(parents=True)
        logging.info(f"{target_path} created")

    t = Timer()
    chronometer = []
    process_names = []

    t.start()
    user_ids = consolidate_users(
        processed_path, target_path, group, overwrite=overwrite, group_names=group_names
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"consolidate_users")

    t.start()
    aggregate_daily_stats(
        data_paths, processed_path, user_ids, target_path, group, overwrite=overwrite
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"aggregate_daily_stats")

    t.start()
    sum_word_frequencies_per_group(
        data_paths, processed_path, target_path, group, overwrite=overwrite
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"sum_word_frequencies_per_group")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "rt.user.id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="retweet",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_retweets")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "quote.user.id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="quote",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_quotes")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "in_reply_to_user_id",
        min_freq=experiment_config["thresholds"].get("edge_weight", 1),
        network_type="reply",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_replies")

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
