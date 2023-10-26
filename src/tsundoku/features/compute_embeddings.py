import toml
import numpy as np
import dask.dataframe as dd
import dask
import click
import logging
import os
import matplotlib
import tqdm

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from scipy.sparse import dok_matrix, save_npz
from dotenv import find_dotenv, load_dotenv

from tsundoku.utils.timer import Timer
from tsundoku.utils.files import read_toml

# BERT TOKENIZER FOR WORD EMBEDDINGS
from transformers import BertTokenizer, BertModel
import torch


PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BETOTokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BETOModel = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)

matplotlib.use("agg")


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--overwrite", type=bool, default=False)
def main(experiment, overwrite):
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

    if not key_folders:
        logging.info(
            "There are no folders with experiment data. Check folder_start and folder_end settings."
        )
        return -1

    logging.info(f"Key Folders: {key_folders}")

    # let's go

    data_base = Path(config["path"]["data"]) / "interim"
    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    if not processed_path.exists():
        processed_path.mkdir(parents=True)
        logging.info(f"{processed_path} created")
    else:
        logging.info(f"{processed_path} exists")

    data_paths = [data_base / key for key in key_folders]


    # We set the timer for all interactions and groups
    t = Timer()
    chronometer = []
    process_names = []

    # matrices
    stopwords_file = Path(config["path"]["config"]) / "stopwords.txt"

    if not stopwords_file.exists():
        stopwords_file = None

    # this file exists from group_users
    elem_to_id = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")["row_id"]
        .compute()
        .to_dict()
    )

    # users embeddings
    t.start()
    group_user_tweets_list(elem_to_id, data_paths, processed_path, overwrite=overwrite)
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"users_embeddings")

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer process names: " + str(process_names))


def dd_from_parquet_paths(paths, min_size=100):
    valid_paths = list(
        filter(lambda x: os.path.exists(x) and os.stat(x).st_size >= min_size, paths)
    )
    return dd.read_parquet(valid_paths)


def group_user_tweets_list(elem_to_id, data_paths, destination_path, overwrite=False):
    user_embedding_target = destination_path / "users.all.embeddings.matrix.npz"

    if not overwrite and user_embedding_target.exists():
        logging.info("tweets lists by user were computed! skipping.")
        return

    def combine_lists(lst):
        combined_lst = []
        for sublist in lst:
            combined_lst.extend(sublist)
        return combined_lst

    tweets_list = (
        dd_from_parquet_paths([d / "tweets_list_per_user.parquet" for d in data_paths])
        .groupby("user.id")["tweets"]
        .apply(combine_lists)  # , meta={"user.id": "int", "tweets": "object"}
        .reset_index()
        .compute()
    )

    embeddings_list = []
    user_id_list = []
    for index, tweet_list in tqdm.tqdm(
        tweets_list.iterrows(), total=tweets_list.shape[0]
    ):
        tweet_list_embedding = []
        for tweet in tweet_list["tweets"]:
            # Tokenize our sentence with the BERT tokenizer.
            tokenized_text = BETOTokenizer.tokenize(tweet)
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = BETOTokenizer.convert_tokens_to_ids(tokenized_text)
            # Display the words with their indeces.
            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            BETOModel.eval()
            with torch.no_grad():
                outputs = BETOModel(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)

            # `token_vecs` is a tensor with shape [n x 768]
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all n token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)

            token_vecs_sum = []
            # For each token in the sentence...
            for token in token_embeddings:
                # Sum the vectors from the last four layers.
                sum_vec = torch.sum(token[-4:], dim=0)
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)
            # `token_vecs` is a tensor with shape [n x 768]
            token_vecs = hidden_states[-2][0]

            # Calculate the average of all n token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            tweet_list_embedding.append(sentence_embedding)

        tweet_list_embedding = torch.stack(tweet_list_embedding)
        user_embedding = torch.mean(tweet_list_embedding, dim=0)
        embeddings_list.append(user_embedding.numpy())
        user_id_list.append(tweet_list["user.id"])

    n = len(embeddings_list[0])
    user_embeddings_matrix = dok_matrix(
        (max(elem_to_id.values()) + 1, n + 1), dtype=np.float32
    )

    for user_id, user_embedding in zip(user_id_list, embeddings_list):
        if not user_id in elem_to_id:
            continue
        matrix_id = elem_to_id[user_id]
        user_embeddings_matrix[matrix_id, :n] = user_embedding

    user_embeddings_matrix = user_embeddings_matrix.tocsr()
    save_npz(user_embedding_target, user_embeddings_matrix)

    logging.info(f"users.embeddings matrix -> {user_embedding_target}")



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
