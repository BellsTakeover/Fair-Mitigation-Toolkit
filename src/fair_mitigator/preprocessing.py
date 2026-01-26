#will read the data: block from YAML
#creates derived feature
#one-hot encodes
#handle missing columns/data
import pandas as pd
import numpy as np

def apply_derived_features(df, derived_features, policy):
    out = df.copy()

    for spec in derived_features or []:
        name = spec["name"]

        if spec["type"] == "expression":
            try:
                out[name] = pd.eval(spec["expr"], engine="python", local_dict=out)
            except Exception as e:
                if policy == "error":
                    raise
                print(f"[WARN] Skipping {name}: {e}")

        elif spec["type"] == "bin":
            src = spec["source"]
            if src not in out.columns:
                if policy == "error":
                    raise ValueError(f"Missing column {src}")
                print(f"[WARN] Missing {src}, skipping {name}")
                continue

            out[name] = pd.cut(out[src], bins=spec["bins"], labels=spec["labels"])
            if spec.get("one_hot", False):
                out = pd.get_dummies(out, columns=[name], drop_first=True)

    return out

def make_features(df, data_cfg):
    target = data_cfg["target_col"]
    y = df[target].astype(int)

    feature_cols = data_cfg.get("feature_cols")
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].copy()

    X = apply_derived_features(
        X,
        data_cfg.get("derived_features", []),
        data_cfg.get("missing_feature_policy", "warn_skip")
    )

    cat_cols = [c for c in data_cfg.get("categorical_cols", []) if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(X.mean())

    return X, y
