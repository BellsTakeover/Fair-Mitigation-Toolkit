from __future__ import annotations

from joblib import dump
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from ..io import save_json, save_csv
from ..metrics import evaluate, cross_validate
from ..models import make_model
from ..preprocessing import make_features
from ..fairness import compute_fairness_report
from ..mitigations.relabel import run_relabeling


def run_smote(df, cfg, outdir):
    rs = int(cfg.get("random_state", 42))

    X_all, y_all, df_raw = make_features(df, cfg["data"])

    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y_all if split_cfg.get("stratify", True) else None,
    )

    #Optional relabeling first (keeps SMOTE from copying bad labels)
    rel_cfg = cfg.get("relabeling", {"enabled": False})
    if rel_cfg.get("enabled", False):
        rel_res = run_relabeling(X_train, y_train, rel_cfg, outdir=outdir)
        y_train_used = rel_res.y_train_cleaned
    else:
        rel_res = None
        y_train_used = y_train

    sm_cfg = cfg.get("smote", {"enabled": True})
    if not sm_cfg.get("enabled", True):
        raise ValueError("smote pipeline expects smote.enabled = true")

    #SMOTE needs at least k+1 samples in the minority class
    minority_count = int(y_train_used.value_counts().min())
    k_req = int(sm_cfg.get("k_neighbors", 5))   
    k_use = min(k_req, max(1, minority_count - 1))

    if k_use < k_req:
        print(f"[WARN] SMOTE k_neighbors reduced from {k_req} to {k_use} (minority_count={minority_count})")

    sm = SMOTE(
        sampling_strategy=sm_cfg.get("sampling_strategy", "auto"),
        k_neighbors=k_use,
        random_state=int(sm_cfg.get("random_state", rs)),
    )

    #X_train_res, y_train_res = sm.fit_resample(X_train, y_train_used)

    #make it's already balanced and SMOE won't change counts
    counts = y_train_used.value_counts()
    if counts.min() == counts.max():
        print("[WARN] Training set is already balanced; skipping SMOTE.")
        X_train_res, y_train_res = X_train, y_train_used
    else:
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train_used)


    model = make_model(cfg["model"])
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    results = evaluate(y_test, y_pred)
    results["pipeline"] = "smote"
    results["train_shape_before_smote"] = [int(X_train.shape[0]), int(X_train.shape[1])]
    results["train_shape_after_smote"] = [int(X_train_res.shape[0]), int(X_train_res.shape[1])]

    if rel_cfg.get("enabled", False) and rel_res is not None:
        results["relabeling_enabled"] = True
        results["relabel_threshold"] = float(rel_cfg.get("confidence_threshold", 0.9))
        results["num_suspicious"] = int(rel_res.suspicious_mask.sum())
    else:
        results["relabeling_enabled"] = False

    save_json(outdir, "metrics.json", results)

    fair_cfg = cfg.get("fairness", {"enabled": False})
    if fair_cfg.get("enabled", False):
        sens_cols = fair_cfg.get("sensitive_cols", [])
        pos_label = int(fair_cfg.get("positive_label", 1))

        fair_df, fair_summary = compute_fairness_report(
            df_raw=df_raw.loc[X_test.index],
            y_true=y_test,
            y_pred=y_pred,
            sensitive_cols=sens_cols,
            positive_label=pos_label,
        )
        save_csv(outdir, "fairness_by_group.csv", fair_df)
        save_json(outdir, "fairness_summary.json", fair_summary)

    cv_cfg = cfg.get("cross_validation", {"enabled": False})
    if cv_cfg.get("enabled", False):
        # CV with SMOTE is trickier to do "perfectly" (needs pipeline per fold),
        # so we keep CV off by default for this pipeline.
        pass

    if cfg.get("outputs", {}).get("save_model", True):
        dump(model, outdir / "model.joblib")

    return results
