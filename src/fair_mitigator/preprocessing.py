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
    if data_cfg is None:
        raise ValueError("Missing 'data:' section in the config file.")

    target = data_cfg["target_col"]
    if target not in df.columns:
        raise ValueError(f"target_col '{target}' not found in CSV columns.")

    #y should be numeric
    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column has non-numeric values or blanks. Clean the target first.")
    y = y.astype(int)

    policy = data_cfg.get("missing_feature_policy", "warn_skip")

    #Drop columns if user requests it
    drop_cols = data_cfg.get("drop_cols", []) or []

    #Decide feature columns
    feature_cols = data_cfg.get("feature_cols", None)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target]
    else:
        # warn/skip missing requested features
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            if policy == "error":
                raise ValueError(f"Missing feature_cols: {missing}")
            print(f"[WARN] Missing feature_cols (skipping): {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns and c != target]

    #Apply drop_cols
    if drop_cols:
        feature_cols = [c for c in feature_cols if c not in drop_cols]

    X = df[feature_cols].copy()

    # Derived features (WHR, BMI bins, etc.)
    X = apply_derived_features(
        X,
        data_cfg.get("derived_features", []),
        policy
    )

    #Decide which columns are categorical
    cat_cfg = data_cfg.get("categorical_cols", None)

    if cat_cfg is None:
        #auto-detect strings --> make sure it doesn't crash
        cat_cols = list(X.select_dtypes(include=["object", "string", "category"]).columns)
    else:
        #user list: warn/skip missing
        missing_cats = [c for c in cat_cfg if c not in X.columns]
        if missing_cats:
            if policy == "error":
                raise ValueError(f"Missing categorical_cols: {missing_cats}")
            print(f"[WARN] Missing categorical_cols (skipping): {missing_cats}")
        cat_cols = [c for c in cat_cfg if c in X.columns]

    #One-hot encode ONCE (only if there is something to encode) --> reduce errors
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    #Force numeric and clean NaNs
    X = X.apply(pd.to_numeric, errors="coerce")

    #Mean-impute numeric columns only
    num_cols = X.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

    #Anything still NaN (all-NaN cols, etc.)
    X = X.fillna(0)

    df_raw_aligned = df.copy()
    return X, y, df_raw_aligned

