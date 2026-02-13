#help for the pipelines/latent.py

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def fit_transform_latent(df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame, spec: dict):
    """
    Fit on train, transform train+test. Returns:
      Z_train_df, Z_test_df, fitted_pipeline
    """

    cat_cols = spec.get("categorical_cols", [])
    num_cols = spec.get("numerical_cols", [])

    # only keep cols that exist (lets the toolkit be forgiving)
    cat_cols = [c for c in cat_cols if c in df_train_raw.columns]
    num_cols = [c for c in num_cols if c in df_train_raw.columns]

    n_components = int(spec.get("n_components", 2))
    prefix = str(spec.get("prefix", "Latent"))

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=str(spec.get("impute_strategy", "mean")))),
        ("scaler", StandardScaler() if bool(spec.get("scale", True)) else "passthrough"),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("pca", PCA(n_components=n_components, random_state=int(spec.get("random_state", 42)))),
    ])

    Z_train = pipe.fit_transform(df_train_raw)
    Z_test = pipe.transform(df_test_raw)

    cols = [f"{prefix}{i+1}" for i in range(n_components)]
    Z_train_df = pd.DataFrame(Z_train, index=df_train_raw.index, columns=cols)
    Z_test_df = pd.DataFrame(Z_test, index=df_test_raw.index, columns=cols)

    return Z_train_df, Z_test_df, pipe
