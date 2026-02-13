# before any pipeline is run this will run in order to make sure the configs are correct
#runs through all possible errors
from __future__ import annotations
import pandas as pd

def validate_data_config(df: pd.DataFrame, data_cfg: dict):
    if data_cfg is None:
        raise ValueError("Missing 'data' block in config.")

    target = data_cfg.get("target_col")
    if not target:
        raise ValueError("data.target_col is required.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")

    drop_cols = data_cfg.get("drop_cols", []) or []

    feat_cols = data_cfg.get("feature_cols", None)
    if feat_cols is not None:
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols contains missing columns: {missing}")
        if target in feat_cols:
            raise ValueError("feature_cols must NOT include target_col.")
        X_cols = [c for c in feat_cols if c not in drop_cols]
    else:
        X_cols = [c for c in df.columns if c != target and c not in drop_cols]

    if len(X_cols) == 0:
        raise ValueError("No features selected. Check feature_cols/drop_cols/target_col.")

    y = df[target].dropna()
    u = set(y.unique())
    if not u.issubset({0, 1, "0", "1"}):
        raise ValueError(f"Target must be binary 0/1. Found: {sorted(list(u))[:10]}")

    return True
