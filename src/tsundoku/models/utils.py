import pandas as pd

def load_model_features(path, model_type, relevant_only=True, users=None):
    importances = pd.read_parquet(
        path / f"{model_type}.classification.features.parquet"
    ).drop("index", axis=1)
    features = (
        pd.read_parquet(
            path / f"{model_type}.classification.term_associations.parquet"
        )
        .rename(columns={"index": "label"})
        .merge(importances)
    )

    if relevant_only:
        features = features[features["xgb.relevance"] > 0].copy()

    def type_names(kind):
        if kind == "tweet_tokens":
            kind = "term"
        elif kind == "description_token":
            kind = "bio"
        elif kind == "user_name":
            kind = "name"
        elif kind.endswith("_group"):
            int_type = kind.split("_")[0].upper()
            kind = f"{int_type}(group)"
        elif kind.endswith("_target"):
            int_type = kind.split("_")[0].upper()
            kind = f"{int_type}"
        elif kind == "profile_domain":
            kind = "bio.link"
        elif kind == "domain":
            kind = "link"
        elif kind == "profile_tld":
            kind = "bio.tld"

        if kind == "MENTION":
            return "REPLY"

        return kind

    features["type"] = features["type"].str.replace(":", "").map(type_names)

    if users is not None:
        feature_tokens = []

        for tup in features.itertuples():
            # print(tup)
            token = getattr(tup, "token")
            kind = getattr(tup, "type")

            if kind in ("RT", "MENTION", "QUOTE", "REPLY"):
                user_id = int(token)
                rows = users[users["user.id"] == user_id]

                if rows.empty:
                    token = f"ID:{token}"
                else:
                    # print(user_id, users[users['user.id'] == user_id].shape)
                    token = "@" + rows["user.screen_name"].values[0]

            feature_tokens.append(token)
            # break

        features["token"] = feature_tokens

    features["new_label"] = (
        features["type"].str.replace(":", "") + ":" + features["token"].astype(str)
    )

    return features
