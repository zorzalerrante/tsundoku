import logging
import os
import re
import zlib
import ahocorasick
import dask
import pandas as pd
import pytz
import toml
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd

from lru import LRU
from cytoolz import pluck
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path

from pyarrow import json
from tsundoku.utils.files import read_list, write_parquet
from tsundoku.utils.text import tokenize
from tsundoku.utils.re import build_re_from_files
from tsundoku.utils.tweets import TWEET_DTYPES, TWEET_DTYPES_RAW


class TweetImporter(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None
        self.logger = logging.getLogger(__name__)

        with open(self.config_file, "rt") as f:
            self.config = toml.load(f)["project"]

        logging.info(self.config)

        self.configure_locations()
        self.configure_language()
        self.configure_terms()
        self.configure_tokenizer()

        self.timezone = pytz.timezone(self.config["content"].get("timezone"))
        dask.config.set(
            pool=ThreadPool(int(self.config["environment"].get("n_jobs", 1)))
        )

    def filter_dataframe(self, df):
        flag = df["id"].notna()

        if not self.location["accept_unknown"]:
            if self.location["patterns"]:
                flag = flag & (
                    df["user.location"].str.contains(self.location["patterns"]) == True
                )

        if self.location["blacklist"]:
            flag = flag & ~(
                df["user.location"].str.contains(self.location["blacklist"]) == True
            )

        candidates = df[flag]
        # self.logger.info(f"Location filtering: {len(candidates)} from {len(df)} tweets")

        if len(self.automaton):
            result = []
            for tuple in candidates.itertuples():
                # print(getattr(tuple, 'text'))
                findings = set(pluck(1, self.automaton.iter(getattr(tuple, "text"))))
                # print(findings)

                if self.terms["patterns"] is not None:
                    # we have keywords:
                    result.append(
                        "search-term" in findings and not "rejected-term" in findings
                    )
                else:
                    result.append(not "rejected-term" in findings)

            candidates = candidates[result]

        # self.logger.info(f"Keyword filtering: {len(candidates)} from {len(df)} tweets")

        return candidates

    def read_tweet_dataframe(self, filename):
        adf = pd.read_parquet(filename, engine="pyarrow")

        if len(adf.columns) != 0:
            return self.filter_dataframe(adf)

        return adf

    def configure_locations(self):
        location_config = self.config["content"].get("location", {})
        self.location = {}
        self.location["accept_unknown"] = bool(location_config.get("accept_unknown", 1))

        if location_config is not None and "blacklist" in location_config:
            blacklist_files = map(
                lambda x: self.config["path"].get("config") + "/" + x,
                location_config["blacklist"],
            )
            patterns = list(
                map(
                    lambda x: x.split(";")[0],
                    filter(lambda x: x, chain(*map(read_list, blacklist_files))),
                )
            )
            self.location["blacklist"] = re.compile("|".join(patterns), re.IGNORECASE)
        else:
            self.logger.warning("no blacklisted locations used")
            self.location["blacklist"] = None

        if location_config is not None and "gazetteers" in location_config:
            gazetteers = map(
                lambda x: self.config["path"].get("config") + "/" + x,
                location_config["gazetteers"],
            )
            patterns = list(
                map(
                    lambda x: x.split(";")[0],
                    filter(lambda x: x, chain(*map(read_list, gazetteers))),
                )
            )
            self.location["patterns"] = re.compile("|".join(patterns), re.IGNORECASE)
        else:
            self.logger.warning("no location filters used")
            self.location["patterns"] = None

    def configure_terms(self):
        self.automaton = ahocorasick.Automaton()

        term_files = self.config["content"].get("term_files", None)
        self.terms = {}

        if term_files is not None:
            term_files = list(
                map(lambda x: self.config["path"].get("config") + "/" + x, term_files)
            )
            self.terms["patterns"] = []  # build_re_from_files(term_files)
            for filename in term_files:
                terms = read_list(filename)
                self.terms["patterns"].extend(terms)
                for term in terms:
                    self.automaton.add_word(term, "search-term")
                self.logger.info(f"read keywords from {filename}: {terms}")
        else:
            self.logger.warning("no keyword terms used")
            self.terms["patterns"] = None

        blacklist = self.config["content"].get("blacklist_files", None)

        if blacklist is None:
            self.logger.warning("no blacklisted keywords used")
            self.terms["blacklist"] = None
        else:
            blacklist = map(
                lambda x: self.config["path"].get("config") + "/" + x, blacklist
            )
            self.terms["blacklist"] = []  # build_re_from_files(blacklist)

            for filename in blacklist:
                terms = read_list(filename)
                self.terms["blacklist"].extend(terms)
                for term in terms:
                    self.automaton.add_word(term, "rejected-term")
                self.logger.info(f"read blacklisted keywords from {filename}: {terms}")

        blacklist_urls = self.config.get("blacklist_urls", None)

        if blacklist_urls is not None:
            blacklist_urls = map(
                lambda x: self.config["path"].get("config") + "/" + x, blacklist_urls
            )
            self.terms["blacklist_urls"] = build_re_from_files(blacklist_urls)
        else:
            self.logger.warning("no blacklisted URLs")

        self.automaton.make_automaton()

    def configure_language(self):
        self.languages = self.config["content"].get("accepted_lang", None)

    def configure_tokenizer(self):
        dtm_config = self.config["content"].get("user_matrix", {})
        ngram_range = dtm_config.get("ngram_range", None)
        stopwords_file = dtm_config.get("stopwords_file", None)

        if stopwords_file is not None:
            stopwords_file = self.config["path"].get("config") + "/" + stopwords_file
            self.logger.info(f"stopwords file: {stopwords_file}")

        if stopwords_file is None:
            stopwords = set()
            self.logger.info("no stopwords")
        else:
            stopwords = set(read_list(stopwords_file))
            self.logger.info(f"#stopwords: {len(stopwords)}")

        token_cache = LRU(dtm_config.get("lru_size", 50))

        def lru_tokenize(x):
            if x in token_cache:
                return token_cache[x]

            result = tokenize(x, ngram_range=ngram_range, stopwords=stopwords)

            token_cache[x] = result

            return result

        self.tokenize = lru_tokenize

    def data_path(self):
        return Path(self.config["path"].get("data"))

    def parse_date_data_to_parquet(
        self, date, pattern, source_path, target_path, periods=24 * 6, freq="10t"
    ):
        date = pd.to_datetime(date)

        if type(source_path) == str:
            source_path = Path(source_path)
        elif not isinstance(source_path, Path):
            raise ValueError(
                f"source_path is not a valid object (Path or str needed, got {type(source_path)})"
            )

        if not source_path.exists():
            raise ValueError(f"source_path ({source_path}) is not a valid path")

        self.logger.info(f"Source folder: {source_path}")

        if not source_path.exists():
            raise IOError(f"{source_path} does not exist")

        data_date = self.timezone.localize(date).astimezone(pytz.utc)
        self.logger.info(f"UTC start date: {data_date}")

        task_files = []

        for date in pd.date_range(data_date, periods=periods, freq=freq):
            file_path = source_path / pattern.format(date.strftime("%Y%m%d%H%M"))

            if not file_path.exists():
                self.logger.info(f"{file_path} does not exist")
            else:
                task_files.append(file_path)

        self.logger.info(f"#files to transform: {len(task_files)}")

        read_tweets = self.import_parquet_files(task_files, target_path)
        return read_tweets

    def import_parquet_files(self, file_names, target_path):
        if not target_path.exists():
            target_path.mkdir(parents=True)
            self.logger.info("{} directory created".format(target_path))
        else:
            self.logger.info("{} exists".format(target_path))

        tasks = [
            dask.delayed(self._parse_files_to_parquet)(i, f, target_path)
            for i, f in enumerate(file_names)
        ]
        read_tweets = sum(dask.compute(*tasks))
        self.logger.info(f"done! imported {read_tweets} tweets from {len(file_names)}")
        return read_tweets

    def _parse_files_to_parquet(self, i, filename, target_path):
        try:
            df = json.read_json(filename)
        except zlib.error:
            self.logger.error(f"ZLIB EXCEPTION - (#{i}) corrupted file: {filename}")
            return 0
        except pa.lib.ArrowInvalid:
            self.logger.error(f"PYARROW EXCEPTION - (#{i}) corrupted file: {filename}")
            return 0

        target_file = target_path / f"{Path(filename).stem}.parquet"

        pq.write_table(df, target_file, use_dictionary=False)
        return df.num_rows

    def import_date(self, date, pattern, source_path, periods=24 * 6, freq="10t"):
        date_str = date
        date = pd.to_datetime(date)

        if type(source_path) == str:
            source_path = Path(source_path)
        elif not isinstance(source_path, Path):
            raise ValueError(
                f"source_path is not a valid object (Path or str needed, got {type(source_path)})"
            )

        if not source_path.exists():
            raise ValueError(f"source_path ({source_path}) is not a valid path")

        self.logger.info(f"Source folder: {source_path}")

        if not source_path.exists():
            raise IOError(f"{source_path} does not exist")

        data_date = self.timezone.localize(date).astimezone(pytz.utc)
        self.logger.info(f"UTC start date: {data_date}")

        task_files = []

        for date in pd.date_range(data_date, periods=periods, freq=freq):
            file_path = source_path / pattern.format(date.strftime("%Y%m%d%H%M"))

            if not file_path.exists():
                self.logger.info(f"{file_path} does not exist")
                pass
            else:
                task_files.append(file_path)

        self.logger.info(f"#files to import: {len(task_files)}")

        parquet_path = self.data_path() / "raw" / date_str

        imported_tweets = self.import_files(
            task_files, parquet_path, file_prefix="tweets.partition"
        )
        return imported_tweets

    def import_files(self, file_names, target_path, file_prefix=None):
        if not target_path.exists():
            target_path.mkdir(parents=True)
            self.logger.info("{} directory created".format(target_path))
        else:
            self.logger.info("{} exists".format(target_path))

        tasks = [
            dask.delayed(self._read_parquet_file)(
                i, f, target_path, file_prefix=file_prefix
            )
            for i, f in enumerate(file_names)
        ]
        read_tweets = sum(dask.compute(*tasks))
        self.logger.info(f"done! imported {read_tweets} tweets")
        return read_tweets

    def _read_parquet_file(self, i, filename, target_path, file_prefix=None):
        df = self.read_tweet_dataframe(filename)

        if file_prefix is not None:
            target_file = target_path / f"{file_prefix}.{i}.parquet"
        else:
            target_file = target_path / f"{Path(filename).stem}.{i}.parquet"

        df["tweet.tokens"] = df["text"].map(self.tokenize)
        df["user.description_tokens"] = df["user.description"].map(self.tokenize)
        df["user.name_tokens"] = df["user.name"].map(self.tokenize)
        # we transform dates from format Sat Jan 01 11:27:55 +0000 2022 to datetime object
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["user.created_at"] = pd.to_datetime(df["user.created_at"])
        write_parquet(df, target_file)
        return len(df)
