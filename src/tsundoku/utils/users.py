import pyarrow as pa


USERS_DTYPES = pa.schema(
    [
        pa.field("user.id", pa.int64()),
        pa.field("row_id", pa.int64()),
        pa.field("user.description", pa.string()),
        pa.field("user.location", pa.string()),
        pa.field("user.name", pa.string()),
        pa.field("user.screen_name", pa.string()),
        pa.field("user.url", pa.string()),
        pa.field("user.protected", pa.bool_()),
        pa.field("user.verified", pa.bool_()),
        pa.field("user.followers_count", pa.int64()),
        pa.field("user.friends_count", pa.int64()),
        pa.field("user.listed_count", pa.int64()),
        pa.field("user.favourites_count", pa.int64()),
        pa.field("user.statuses_count", pa.int64()),
        pa.field("user.created_at", pa.timestamp("ns", tz="UTC")),
        pa.field("user.profile_image_url_https", pa.string()),
        pa.field("user.default_profile", pa.bool_()),
        pa.field("user.default_profile_image", pa.bool_()),
        pa.field("user.description_tokens", pa.list_(pa.string())),
        pa.field("user.name_tokens", pa.list_(pa.string())),
        pa.field("user.dataset_tweets", pa.int64()),
        pa.field("predicted.stance", pa.string()),
        pa.field("predicted.person", pa.string()),
        pa.field("__null_dask_index__", pa.int64()),
    ]
)
