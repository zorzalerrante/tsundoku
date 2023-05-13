# -*- coding: utf-8 -*-
import logging
import gzip
import rapidjson as json

from dotenv import find_dotenv, load_dotenv
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
import gensim
from scipy.sparse import load_npz
from cytoolz import keymap, valmap

from tsundoku.helpers import read_toml


@click.command()
@click.option("--experiment", type=str, default="full")
def main(experiment):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    experiment_name = experiment
    # group_key = group

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(
        Path(os.environ['TSUNDOKU_PROJECT_PATH']) / "config.toml")["project"]
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

    # with open(Path(config["path"]["config"]) / "groups" / f"{group_key}.toml") as f:
    #    group_config = toml.load(f)

    users = pd.read_json(
        processed_path / "consolidated" / "user.consolidated_groups.parquet", lines=True
    )
    users.head()

    docterm_matrix = load_npz(processed_path / "user.tweets.matrix.npz")

    vocabulary = pd.read_json(
        processed_path / "user.tweet_vocabulary.relevant.parquet", lines=True
    )

    frequencies = pd.read_json(
        processed_path / "consolidated" / "tweet.word_frequencies.parquet", lines=True
    )

    frequent_vocabulary = vocabulary.join(
        frequencies.groupby("token")["n_users"].sum(), on="token", how="inner"
    ).pipe(
        lambda x: x[
            x["n_users"].between(
                experimental_settings["topic_modeling"].get("min_users", 1),
                x["n_users"].quantile(
                    experimental_settings["topic_modeling"].get(
                        "max_users_quantile", 1.0
                    )
                ),
            )
        ]
    )

    print(frequent_vocabulary.head(15))

    filtered_users = users[
        users["user.dataset_tweets"].between(experimental_settings["topic_modeling"].get(
            "min_tweets", 1), users['user.dataset_tweets'].quantile(experimental_settings["topic_modeling"].get("max_tweets_quantile", 1.0)))
    ]

    filtered_docterm_matrix = docterm_matrix[filtered_users["row_id"].values, :][
        :, frequent_vocabulary["token_id"].values
    ]

    corpus = gensim.matutils.Sparse2Corpus(
        filtered_docterm_matrix, documents_columns=False
    )

    lda = gensim.models.ldamulticore.LdaMulticore(
        corpus,
        num_topics=experimental_settings["topic_modeling"].get("n_topics", 50),
        id2word=frequent_vocabulary.reset_index()["token"].to_dict(),
        workers=experimental_settings["topic_modeling"].get("n_jobs", 1),
        passes=experimental_settings["topic_modeling"].get("passes", 1),
        alpha=experimental_settings["topic_modeling"].get("alpha", "symmetric"),
        iterations=experimental_settings["topic_modeling"].get("iterations", 50),
        random_state=experimental_settings["topic_modeling"].get("random_state", 666),
    )

    print(lda.print_topics())

    lda.save(str(processed_path / "consolidated" / f"user.topic_model.gensim.gz"))

    frequent_vocabulary.rename(
        {"index": "topic_term_id", "token_id": "dtm_col_id"}, axis=1
    ).assign(row_id=np.arange(len(frequent_vocabulary))).drop(
        ["frequency", "n_users"], axis=1
    ).to_json(
        processed_path / "consolidated" / "user.topic_model.vocabulary.parquet",
        compression="gzip",
        orient="records",
        lines=True,
    )

    with gzip.open(
        processed_path / "consolidated" / "user.topic_model.doc_topics.parquet", "wt"
    ) as f:
        for i, (doc, uid) in enumerate(
            zip(
                gensim.matutils.Sparse2Corpus(
                    docterm_matrix[:, frequent_vocabulary["token_id"].values],
                    documents_columns=False,
                ),
                users["user.id"].values,
            )
        ):
            record = {"row_id": i, "user.id": int(uid)}
            record.update(
                valmap(float, keymap(str, dict(lda.get_document_topics(doc))))
            )
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
