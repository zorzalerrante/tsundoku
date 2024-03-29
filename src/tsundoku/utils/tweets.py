import pyarrow as pa

TWEET_FIELDS = [
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("created_at", pa.timestamp("ns", tz="UTC")),
    pa.field("lang", pa.string()),
    pa.field(
        "entities",
        pa.struct(
            {
                "urls": pa.list_(
                    pa.struct(
                        {
                            "url": pa.string(),
                            "expanded_url": pa.string(),
                            "display_url": pa.string(),
                        }
                    )
                ),
                "user_mentions": pa.list_(
                    pa.struct(
                        {
                            "screen_name": pa.string(),
                            "name": pa.string(),
                            "id": pa.int64(),
                        }
                    )
                ),
                "hashtags": pa.list_(pa.string()),
            }
        ),
    ),
    pa.field(
        "user",
        pa.struct(
            {
                "id": pa.int64(),
                "description": pa.string(),
                "location": pa.string(),
                "name": pa.string(),
                "screen_name": pa.string(),
                "url": pa.string(),
                "protected": pa.bool_(),
                "verified": pa.bool_(),
                "followers_count": pa.int64(),
                "friends_count": pa.int64(),
                "listed_count": pa.int64(),
                "favourites_count": pa.int64(),
                "statuses_count": pa.int64(),
                "created_at": pa.timestamp("ns", tz="UTC"),
                "profile_image_url_https": pa.string(),
                "default_profile": pa.bool_(),
                "default_profile_image": pa.bool_(),
            }
        ),
    ),
    pa.field("is_retweet", pa.bool_()),
    pa.field("is_quote", pa.bool_()),
    pa.field("is_reply", pa.bool_()),
    pa.field("in_reply_to_user_id", pa.int64()),
    pa.field("in_reply_to_status_id", pa.int64()),
    pa.field(
        "quote",
        pa.struct({"id": pa.int64(), "user": pa.struct({"id": pa.int64()})}),
    ),
    pa.field(
        "rt", pa.struct({"id": pa.int64(), "user": pa.struct({"id": pa.int64()})})
    ),
    pa.field("tweet", pa.struct({"tokens": pa.list_(pa.string())})),
]

TWEET_FIELDS_EXTENDED = TWEET_FIELDS.copy()

# Extender TWEET_FIELDS_EXTENDED con los campos adicionales
TWEET_FIELDS_EXTENDED.extend(
    [
        pa.field("user.description_tokens", pa.list_(pa.string())),
        pa.field("user.name_tokens", pa.list_(pa.string())),
    ]
)

TWEET_DTYPES = pa.schema(TWEET_FIELDS_EXTENDED)
TWEET_DTYPES_RAW = pa.schema(TWEET_FIELDS)


def flatten_tweet(tweet):
    text = tweet_text(tweet)
    urls = tweet_urls(tweet)
    mentions = tweet_mentions(tweet)
    hashtags = tweet_hashtags(tweet)

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
        # 'user.profile_image_url': user['profile_image_url'],
        "user.profile_image_url_https": user["profile_image_url_https"]
        if user["profile_image_url_https"]
        else "",
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
