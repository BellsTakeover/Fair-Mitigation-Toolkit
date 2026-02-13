from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class LatentSpec:
    enabled: bool = False
    n_components: int = 3
    prefix: str = "Latent"
    categorical_cols: Optional[List[str]] = None
    numerical_cols: Optional[List[str]] = None
    mode: str = "original_plus_latent"  # or "latent_only"
    impute_strategy: str = "mean"
    scale: bool = True


def _latent_columns(prefix: str, k: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(1, k + 1)]


def build_latent_pipeline(spec: LatentSpec) -> Pipeline:
    cat_cols = spec.categorical_cols or []
    num_cols = spec.numerical_cols or []

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    num_steps = [("imputer", SimpleImputer(strategy=spec.impute_strategy))]
    if spec.scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # PCA expects numeric array
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("pca", PCA(n_components=int(spec.n_components), random_state=42)),
    ])
    return pipe


def fit_transform_latent(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    spec: LatentSpec,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    pipe = build_latent_pipeline(spec)

    Z_train = pipe.fit_transform(X_train_raw)
    Z_test = pipe.transform(X_test_raw)

    cols = _latent_columns(spec.prefix, Z_train.shape[1])
    Z_train_df = pd.DataFrame(Z_train, index=X_train_raw.index, columns=cols)
    Z_test_df = pd.DataFrame(Z_test, index=X_test_raw.index, columns=cols)

    return Z_train_df, Z_test_df, pipe