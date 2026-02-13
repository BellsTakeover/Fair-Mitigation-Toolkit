import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate

# --- Core rate helpers (binary classification) ---

def _safe_div(a, b):
    return float(a) / float(b) if b else np.nan

def tpr(y_true, y_pred, pos_label=1):
    # True Positive Rate = TP / (TP + FN)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
    return _safe_div(tp, tp + fn)

def fpr(y_true, y_pred, pos_label=1):
    # False Positive Rate = FP / (FP + TN)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    tn = int(((y_true != pos_label) & (y_pred != pos_label)).sum())
    return _safe_div(fp, fp + tn)

def compute_fairness_report(df_raw, y_true, y_pred, sensitive_cols, positive_label=1):
    """
    Returns:
      - fairness_by_group_df: long-form table with per-group metrics for each sensitive feature
      - fairness_summary: dict with max-min gaps per sensitive feature (researcher-friendly)
    Metrics:
      - accuracy
      - selection_rate (P(ŷ=1))  -> used for Demographic Parity gap
      - tpr (Equal Opportunity)
      - fpr (part of Equalized Odds)
    """
    rows = []
    summary = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metric_fns = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,  # fairlearn metric: mean(y_pred==1)
        "tpr": lambda yt, yp: tpr(yt, yp, pos_label=positive_label),
        "fpr": lambda yt, yp: fpr(yt, yp, pos_label=positive_label),
    }

    for col in sensitive_cols:
        if col not in df_raw.columns:
            continue

        sens = df_raw[col].astype("string").fillna("MISSING")

        mf = MetricFrame(
            metrics=metric_fns,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sens
        )

        # Per-group table
        by_group = mf.by_group.reset_index().rename(columns={"index": "group"})
        by_group.insert(0, "sensitive_feature", col)
        rows.append(by_group)

        # Researcher summary: gaps (max-min) ignoring NaNs
        gaps = {}
        for m in ["accuracy", "selection_rate", "tpr", "fpr"]:
            s = mf.by_group[m].astype(float)
            s = s[~s.isna()]
            gaps[m + "_gap"] = float(s.max() - s.min()) if len(s) else np.nan

        summary[col] = {
            "demographic_parity_gap": gaps["selection_rate_gap"],
            "equal_opportunity_gap": gaps["tpr_gap"],
            "equalized_odds_gaps": {
                "tpr_gap": gaps["tpr_gap"],
                "fpr_gap": gaps["fpr_gap"],
            },
            "accuracy_gap": gaps["accuracy_gap"],
        }

    if rows:
        fairness_by_group_df = pd.concat(rows, ignore_index=True)
    else:
        fairness_by_group_df = pd.DataFrame(
            columns=["sensitive_feature", "group", "accuracy", "selection_rate", "tpr", "fpr"]
        )

    return fairness_by_group_df, summary
