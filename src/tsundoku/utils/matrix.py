import dask.dataframe as dd
from scipy.sparse import load_npz


def load_matrix_and_features(
    data_path, matrix_key, names_key, name, index="token", token_id="token_id"
):
    if not names_key.endswith("parquet"):
        names_key = f"{names_key}.relevant.parquet"

    raw_matrix = load_npz(data_path / f"{matrix_key}.matrix.npz")
    raw_features = dd.read_parquet(data_path / names_key)

    if index != "token":
        raw_features = raw_features.rename(columns={index: "token"})

    if token_id != "token_id":
        raw_features = raw_features.rename(columns={token_id: "token_id"})

    raw_features["type"] = name

    print(repr(raw_matrix))
    print(raw_features.head())

    return raw_matrix, raw_features
