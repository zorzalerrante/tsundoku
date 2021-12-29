# 📚 tsundoku

Tsundoku is a Python toolkit to analyze Twitter data, following the methodology published in:

> Graells-Garrido, E., Baeza-Yates, R., & Lalmas, M. (2020, July). [Every colour you are: Stance prediction and turnaround in controversial issues](https://dl.acm.org/doi/abs/10.1145/3394231.3397907). In 12th ACM Conference on Web Science (pp. 174-183).

About the name: tsundoku is a Japanese word (積ん読) that means "to pile books without reading them" (see more in [Wikipedia](https://en.wikipedia.org/wiki/Tsundoku)). It is common to crawl tweets and do nothing with them later. So `tsundoku` provides a way to work with all those piled-up tweets.

## Development Setup

```sh
# Create conda environment, install dependencies on it and activate it
conda create --name tsundoku --file environment.yml
conda activate tsundoku

python -m ipykernel install --user --name tsundoku --display-name "Python (tsundoku)"
```

## Data preliminaries

### Global configuration and preprocessing

Create an `.env` file in the root of this repository with the following structure:

```
TSUNDOKU_PROJECT_PATH=./example_project
INCOMING_PATH=/home/egraells/data/tweets/incoming
TWEET_PATH=/home/egraells/data/tweets/2021_flattened
TSUNDOKU_LANGUAGES="es|und"
```

This code assumes that you crawl tweets using the Streaming API. These tweets are stored in JSON format, one tweet per line, in files compressed using gzip, stored in the  `INCOMING_PATH` folder.

The `TWEET_PATH` folder is where the system stores a first pre-processed version of tweets from `INCOMING_PATH`. In this first step, tsundoku does two things: first, it keeps tweets in the specified languages; second, it flattens the tweet structure and removes some unused attributes. It does this through the following command:

```bash
$ python -m tsundoku.data.filter_and_flatten
```

Note that this operation deletes the original files.

### Configure your project

The `TSUNDOKU_PROJECT_PATH` folder defines a project. It contains the following files and folders:

- `config.toml`: project configuration.
- `groups/*.toml`: classifier configuration for several groups of users. This is arbitrary, you can define your own groups. The mandatory one is called `relevant.toml`.
- `experiments.toml`: experiment definition and classifier hyper-parameters. Experiments enable analysis in different periods (for instance, first and second round of a presidential election).
- `keywords.txt` (optional): set of keywords to filter tweets. For instance, presidential candidate names, relevant hashtags, etc.
- `stopwords.txt` (optional): list of stop words.

Please see the example in the `example_project` folder.

In `config.toml` there are two important paths to configure:

```toml
[project.path]
config = "/home/egraells/repositories/tsundoku/example_project"
data = "/home/egraells/repositories/tsundoku/example_project/data"
```

The first path, `config`, states where the project lies. The second path, `data`, states where the imported data will be stored. This includes the raw data and the results from processing.

### Import data into your project

`tsundoku` has three folders within the project data folder: `raw`, `interim`, and `processed`.

The `raw` folder contains a subfolder named `json`, and within `raw/json` there is one folder for each day. The format is `YYYY-MM-DD`. Actually, the name of each folder within `raw/json` could be anything, but by convention I have worked with dates, as it makes it easier to organize different experiments.

Currently, there are two ways of importing data. First, by specifying a chunk of tweet files to be imported into one folder within `raw/json` (A); or second, by importing files when the filename encodes datetime structure (B). Both are described next.

If none of these two options works for you, you will have to craft your own importer. Fortunately, the module `tsundoku.data.importer` contains the `TweetImporter` class that will help you do so. 

#### A. Import a set of files into a specific target

The following command imports a set of files into a specific target folder:

```sh
$ python -m tsundoku.data.import_files /mnt/storage/tweets/*.gz --target 2021-12-12
```

This command takes all files pointed by the wildcard (you can also point specific files) and then it filters the tweets relevant for the project, saving them in a folder named `2021-12-12` in the project. The files do not need to be inside `TWEET_PATH`. However, they do need to be flattened according to the `tsundoku.data.filter_and_flatten` script. 

#### B. Import by date when files have a specific naming structure

The following command imports a specific date from `TWEET_PATH`:

```sh
$ python -m tsundoku.data.import_date 20211219
```

This command assumes that tweet files have a specific file naming schema, although this is not a requirement. The schema is the following:

`auroracl_202112271620.data.gz`

Where:

* `auroracl_` is an optional prefix. In this case, it's the codename of the project that started this repository a few years ago.
* `2021` (year) `12` (month) `27` (day) `1620` (time of the day).

The code I use to crawl tweets generates these files every 10 minutes. It is available in [this repository](https://github.com/zorzalerrante/aguaite).

## Run your project

Let's assume you have already imported the data, and that you have defined at least one experiment. We will run the following commands to perform the experiments:

1. `$ python -m tsundoku.features.compute_features`: this will estimate features (such as document-term matrices) for every day in your project.
2. `$ python -m tsundoku.features.prepare_experiment --experiment experiment_name`: this will prepare the features for the specific experiment. For instance, a experiment has start/end dates, so it consolidates the data between those dates only.
3. `$ python -m tsundoku.models.predict_groups --experiment experiment_name --group relevance`: this command predicts whether a user profile is relevant or not (noise) for the experiment. It uses a XGB classifier.
4. `$ python -m tsundoku.models.predict_groups --experiment experiment_name --group another_group`: this command predicts groups within users. Current sample configurations include _stance_ (which candidate is supported by this profile?), _person_ (sex or institutional account), _location_ (the different regions in Chile). You can define as many groups as you want. Note that for each group you must define categories in the corresponding `.toml` file. In this file, if a category is called _noise_, it means that users who fall in the category will be discarding when consolidating results.
5. `$ python -m tsundoku.analysis.analyze_groups --experiment experiment_name --group reference_group`: this command takes the result from the classification and consolidates the analysis with respect to interaction networks, vocabulary, and other features. It requires a reference group to base the analysis (for instance, _stance_ allows you to characterize the supporters of each presidential candidate).

After this, in your project data folder `data/processed/experiment_name/consolidated` you will find several files with the results of the analysis.
