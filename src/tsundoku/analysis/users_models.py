import pyarrow as pa

USERS_DTYPES = pa.schema([
    ("user.id", pa.int64()),
    ("row_id", pa.int64()),
    ("user.description", pa.string()),
    ("user.location", pa.string()),
    ("user.name", pa.string()),
    ("user.screen_name", pa.string()),
    ("user.url", pa.string()),
    ("user.protected", pa.bool_()),
    ("user.verified", pa.bool_()),
    ("user.followers_count", pa.int64()),
    ("user.friends_count", pa.int64()),
    ("user.listed_count", pa.int64()),
    ("user.favourites_count", pa.int64()),
    ("user.statuses_count", pa.int64()),
    ("user.created_at", pa.timestamp('ns', tz='UTC')),
    ("user.profile_image_url_https", pa.string()),
    ("user.default_profile", pa.bool_()),
    ("user.default_profile_image", pa.bool_()),
    ("user.description_tokens", pa.list_(pa.string())),
    ("user.name_tokens", pa.list_(pa.string())),
    ("user.dataset_tweets", pa.int64()),
    ("predicted.stance", pa.string()),
    ("predicted.person", pa.string()),
    ("__null_dask_index__", pa.int64())
])
