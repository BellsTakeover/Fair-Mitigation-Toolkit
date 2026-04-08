from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_fairness_report(
    df_raw: pd.DataFrame,
    y_true,
    y_pred,
    sensitive_cols: list[str],
    positive_label: int = 1,
):
    """
    Returns:
      fair_df: per-group fairness table
      fair_summary: summary dict by sensitive attribute

    fair_summary structure:
    {
      "sex": {
        "demographic_parity_difference": 0.12,
        "equal_opportunity_difference": 0.08,
        "equalized_odds_difference": 0.10
      },
      ...
    }
    """
    fair_rows = []
    per_attr = {}

    for col in sensitive_cols:
        if col not in df_raw.columns:
            continue

        s = df_raw[col]

        # Per-group summaries
        mf = MetricFrame(
            metrics={"tpr": true_positive_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=s,
        )

        by_group = mf.by_group

        for group_name, row in by_group.iterrows():
            fair_rows.append(
                {
                    "sensitive_attribute": col,
                    "group": str(group_name),
                    "true_positive_rate": float(row["tpr"]),
                }
            )

        # Group disparity summaries
        dp_diff = demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=s,
        )
        eo_diff = MetricFrame(
            metrics=true_positive_rate,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=s,
        ).difference(method="between_groups")
        eod_diff = equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=s,
        )

        per_attr[col] = {
            "demographic_parity_difference": float(dp_diff),
            "equal_opportunity_difference": float(eo_diff),
            "equalized_odds_difference": float(eod_diff),
        }

    fair_df = pd.DataFrame(fair_rows)
    return fair_df, per_attr


def save_fairness_plots(fair_df: pd.DataFrame, outdir: Path) -> None:
    """
    Saves simple TPR-by-group bar plots for each sensitive attribute.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if fair_df.empty:
        return

    for attr in fair_df["sensitive_attribute"].dropna().unique():
        sub = fair_df[fair_df["sensitive_attribute"] == attr].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(8, 4))
        plt.bar(sub["group"].astype(str), sub["true_positive_rate"])
        plt.xlabel("Group")
        plt.ylabel("True Positive Rate")
        plt.title(f"Fairness by group: {attr}")
        plt.tight_layout()
        plt.savefig(outdir / f"fairness_{attr}.png", dpi=150)
        plt.close()


def save_fairness_outputs(fair_df: pd.DataFrame, fair_summary: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fair_df.to_csv(outdir / "fairness_by_group.csv", index=False)
    save_json(outdir / "fairness_summary.json", fair_summary)
    save_fairness_plots(fair_df, outdir)
