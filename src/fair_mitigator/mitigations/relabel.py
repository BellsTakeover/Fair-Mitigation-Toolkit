from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..models import make_model


@dataclass
class RelabelResult:
    y_train_cleaned: pd.Series
    suspicious_mask: pd.Series


def run_relabeling(X_train, y_train, rel_cfg: dict, outdir: Path | None = None) -> RelabelResult:
    """
    Train a quick RF on the training set and flag rows where:
      - model prediction != label
      - model is confident (probability > threshold)

    If auto_replace is on, we swap the label to the model's prediction.
    """
    thr = float(rel_cfg.get("confidence_threshold", 0.9))
    auto_replace = bool(rel_cfg.get("auto_replace", True))

    #keep y aligned with X --> prevents index issues
    y_train = pd.Series(y_train, index=X_train.index)

    #use a small-ish RF for speed unless user overrides it
    model_cfg = rel_cfg.get(
        "model",
        {"kind": "random_forest", "params": {"n_estimators": 100, "random_state": 42, "n_jobs": 1}},
    )
    clf = make_model(model_cfg)
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_train)
    y_conf = clf.predict_proba(X_train).max(axis=1)

    # build suspicious mask and keep it indexed
    suspicious = (y_hat != y_train.values) & (y_conf > thr)
    suspicious = pd.Series(suspicious, index=X_train.index)

    y_clean = y_train.copy()
    if auto_replace and suspicious.any():
        y_clean.loc[suspicious] = y_hat[suspicious.values]

    #Save a log if we have an output folder
    if outdir is not None:
        log = pd.DataFrame(
            {
                "row_index": X_train.index,
                "true_label": y_train.values,
                "predicted_label": y_hat,
                "model_confidence": y_conf,
                "suspicious": suspicious,
            }
        ).set_index("row_index")

        log["replaced"] = False
        log["new_label"] = y_train.values
        if auto_replace and suspicious.any():
            log.loc[suspicious, "replaced"] = True
            log.loc[suspicious, "new_label"] = y_hat[suspicious.values]

        log.to_csv(outdir / "relabeling_replacement_log.csv")

    return RelabelResult(y_train_cleaned=y_clean, suspicious_mask=suspicious)
