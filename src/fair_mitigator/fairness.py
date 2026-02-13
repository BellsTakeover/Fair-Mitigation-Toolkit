from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)


def compute_fairness_report(
    df_raw: pd.DataFrame,
    y_true,
    y_pred,
    sensitive_cols: List[str],
    positive_label: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fairness metrics on the test set (after mitigations).

    Rules:
      - sensitive_cols must refer to RAW columns in the CSV (not one-hot).
      - If a column is missing, warn + skip it.
      - Always returns (df, summary). They may be empty if nothing usable exists.
    """
    if not sensitive_cols:
        return pd.DataFrame(), {}

    # Align series to df_raw index
    y_true = pd.Series(y_true, index=df_raw.index)
    y_pred = pd.Series(y_pred, index=df_raw.index)

    rows = []
    summary: Dict[str, Dict] = {}

    for col in sensitive_cols:
        if col not in df_raw.columns:
            print(f"[WARN] fairness: '{col}' not found in raw data. Skipping.")
            continue

        A = df_raw[col].copy()

        # If it's entirely missing, skip it
        if pd.isna(A).all():
            print(f"[WARN] fairness: '{col}' is all NaN. Skipping.")
            continue

        # Keep it simple and stable
        A = A.astype("string").fillna("MISSING")

        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=A,
        )

        by_group = mf.by_group.reset_index().rename(columns={"index": "group"})
        by_group.insert(0, "sensitive_col", col)
        rows.append(by_group)

        summary[col] = {
            "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred, sensitive_features=A)),
            "equal_opportunity_difference": float(equal_opportunity_difference(y_true, y_pred, sensitive_features=A)),
            "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=A)),
        }

    if not rows:
        return pd.DataFrame(), {}

    fairness_df = pd.concat(rows, axis=0, ignore_index=True)
    return fairness_df, summary
