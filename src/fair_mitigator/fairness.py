import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import MetricFrame

def compute_fairness_report(df_raw, y_true, y_pred, sensitive_cols):
    rows = []

    for col in sensitive_cols:
        sens = df_raw[col].astype("string").fillna("MISSING")

        mf = MetricFrame(
            metrics={"accuracy": accuracy_score},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sens
        )

        by_group = mf.by_group.reset_index().rename(columns={"index": "group"})
        by_group.insert(0, "sensitive_feature", col)
        rows.append(by_group)

    return pd.concat(rows, ignore_index=True)
