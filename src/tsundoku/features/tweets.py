import numpy as np
import pandas as pd

from tsundoku.features.re import PUNCTUATION_RE, URL_RE

TWEET_DTYPES = {
    "id": np.dtype("int64"),
    "text": np.dtype("O"),
    "created_at": np.dtype(
        "datetime64[ns]"
    ),  # pd.core.dtypes.dtypes.DatetimeTZDtype(tz='UTC'),
    "lang": np.dtype("O"),
    "entities.urls": np.dtype("O"),
    "entities.user_mentions": np.dtype("O"),
    "entities.hashtags": np.dtype("O"),
    "user.id": np.dtype("int64"),
    "user.description": np.dtype("O"),
    "user.location": np.dtype("O"),
    "user.name": np.dtype("O"),
    "user.screen_name": np.dtype("O"),
    "user.url": np.dtype("O"),
    "user.protected": np.dtype("bool"),
    "user.verified": np.dtype("bool"),
    "user.followers_count": np.dtype("int64"),
    "user.friends_count": np.dtype("int64"),
    "user.listed_count": np.dtype("int64"),
    "user.favourites_count": np.dtype("int64"),
    "user.statuses_count": np.dtype("int64"),
    "user.created_at": np.dtype(
        "datetime64[ns]"
    ),  # pd.core.dtypes.dtypes.DatetimeTZDtype(tz='UTC'),
    #'user.profile_image_url': np.dtype('O'),
    "user.profile_image_url_https": np.dtype("O"),
    "user.default_profile": np.dtype("bool"),
    "user.default_profile_image": np.dtype("bool"),
    "is_retweet": np.dtype("bool"),
    "is_quote": np.dtype("bool"),
    "is_reply": np.dtype("bool"),
    "in_reply_to_user_id": np.dtype("int64"),
    "in_reply_to_status_id": np.dtype("int64"),
    "quote.id": np.dtype("int64"),
    "quote.user.id": np.dtype("int64"),
    "rt.id": np.dtype("int64"),
    "rt.user.id": np.dtype("int64"),
    "tweet.tokens": np.dtype("O"),
    "user.description_tokens": np.dtype("O"),
    "user.name_tokens": np.dtype("O"),
}


def flatten_tweet(tweet):
    text = tweet_text(tweet)
    urls = tweet_urls(tweet)
    mentions = tweet_mentions(tweet)
    hashtags = tweet_hashtags(tweet)
    # created_at = parse_twitter_date(tweet['created_at'], self.timezone)
    # user_created_at = parse_twitter_date(tweet['user']['created_at'], self.timezone)

    user = tweet["user"]

    row = {
        "id": tweet["id"],
        "text": text,
        "created_at": tweet["created_at"],
        "lang": tweet["lang"] if "lang" in tweet else "und",
        # entities
        "entities.urls": "|".join(urls) if urls else "",
        "entities.user_mentions": "|".join(map(str, mentions)) if mentions else "",
        "entities.hashtags": "|".join(hashtags) if hashtags else "",
        # user
        "user.id": user["id"],
        "user.description": user["description"] if user["description"] else "",
        "user.location": user["location"] if user["location"] else "",
        "user.name": user["name"] if user["name"] else "",
        "user.screen_name": user["screen_name"],
        "user.url": user["url"] if user["url"] else "",
        "user.protected": False,
        "user.verified": user["verified"],
        "user.followers_count": user["followers_count"],
        "user.friends_count": user["friends_count"],
        "user.listed_count": user["listed_count"],
        "user.favourites_count": user["favourites_count"],
        "user.statuses_count": user["statuses_count"],
        "user.created_at": user["created_at"],
        #'user.profile_image_url': user['profile_image_url'],
        "user.profile_image_url_https": user["profile_image_url_https"],
        "user.default_profile": user["default_profile"],
        "user.default_profile_image": user["default_profile_image"],
        # other
        "is_retweet": "retweeted_status" in tweet,
        "is_quote": "quoted_status" in tweet,
        "is_reply": tweet["in_reply_to_user_id"] is not None,
        "in_reply_to_user_id": tweet["in_reply_to_user_id"]
        if tweet["in_reply_to_user_id"] is not None
        else 0,
        "in_reply_to_status_id": tweet["in_reply_to_status_id"]
        if tweet["in_reply_to_status_id"] is not None
        else 0,
    }

    if "quoted_status" in tweet and tweet["quoted_status"]:
        quote = tweet["quoted_status"]
        row["quote.id"] = quote["id"]
        row["quote.user.id"] = quote["user"]["id"]
    else:
        row["quote.id"] = 0
        row["quote.user.id"] = 0

    if "retweeted_status" in tweet and tweet["retweeted_status"]:
        rt = tweet["retweeted_status"]
        row["rt.id"] = rt["id"]
        row["rt.user.id"] = rt["user"]["id"]
    else:
        row["rt.id"] = 0
        row["rt.user.id"] = 0

    return row


def tweet_text(tweet):
    if "retweeted_status" in tweet:
        return tweet_text(tweet["retweeted_status"])

    if "extended_tweet" in tweet:
        return tweet["extended_tweet"]["full_text"]
    else:
        return tweet["text"]


def tweet_mentions(tweet):
    if "retweeted_status" in tweet:
        return tweet_mentions(tweet["retweeted_status"])

    people = []

    if "extended_tweet" in tweet:
        entities = tweet["extended_tweet"]["entities"]
    else:
        entities = tweet["entities"]

    for u in entities["user_mentions"]:
        if u["id"]:
            people.append(u["id"])

    if "quoted_status" in tweet and tweet["quoted_status"]:
        people.append(tweet["quoted_status"]["user"]["id"])

    return people


def tweet_hashtags(tweet):
    if "retweeted_status" in tweet:
        return tweet_hashtags(tweet["retweeted_status"])

    hashtags = []

    if "extended_tweet" in tweet:
        entities = tweet["extended_tweet"]["entities"]
    else:
        entities = tweet["entities"]

    for u in entities["hashtags"]:
        if u["text"]:
            hashtags.append(u["text"])

    return hashtags


def tweet_urls(tweet):
    if "retweeted_status" in tweet:
        return tweet_urls(tweet["retweeted_status"])

    urls = []

    if "extended_tweet" in tweet:
        entities = tweet["extended_tweet"]["entities"]
    else:
        entities = tweet["entities"]

    for l in entities["urls"]:
        if l["expanded_url"]:
            urls.append(l["expanded_url"])

    return urls
