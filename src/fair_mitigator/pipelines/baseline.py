from __future__ import annotations

from joblib import dump
from sklearn.model_selection import train_test_split

from fair_mitigator.metrics import evaluate
from fair_mitigator.io import save_json, save_csv
from fair_mitigator.preprocessing import make_features
from fair_mitigator.models import make_model, run_grid_search
from fair_mitigator.fairness import compute_fairness_report

#OPTIONAL: preprocessing mitigation if wanting to do pre- + in-
from fair_mitigator.mitigations.relabel import run_relabeling

def run_baseline(df, cfg, outdir):
    if cfg is None:
        raise ValueError("Configuration file could not be loaded or is empty.")

    rs = int(cfg.get("random_state", 42))

    #Build numeric X and aligned raw df
    X, y, df_raw = make_features(df, cfg["data"])

    # Split (with safe defaults)
    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y if split_cfg.get("stratify", True) else None
    )

    #Optional relabeling (pre-processing) --> multiple pre-processing
    rel_cfg = cfg.get("relabeling", {"enabled": False})
    if rel_cfg.get("enabled", False):
        rel_res = run_relabeling(X_train, y_train, rel_cfg, outdir=outdir)
        y_train_used = rel_res.y_train_cleaned
    else:
        rel_res = None
        y_train_used = y_train

    # Model + optional grid search
    model = make_model(cfg["model"])

    grid_cfg = cfg.get("grid_search", {"enabled": False})
    if grid_cfg.get("enabled", False):
        model, best_params = run_grid_search(model, grid_cfg, X_train, y_train_used)
    else:
        best_params = None
        model.fit(X_train, y_train_used)

    #Predict
    y_pred = model.predict(X_test)

    #Metrics
    metrics = evaluate(y_test, y_pred)
    metrics["pipeline"] = "baseline"
    metrics["model_kind"] = cfg.get("model", {}).get("kind", None)

    if best_params is not None:
        metrics["best_params"] = best_params

    if rel_cfg.get("enabled", False) and rel_res is not None:
        metrics["relabeling_enabled"] = True
        metrics["relabel_threshold"] = float(rel_cfg.get("confidence_threshold", 0.9))
        metrics["num_suspicious"] = int(rel_res.suspicious_mask.sum())
    else:
        metrics["relabeling_enabled"] = False

    metrics["random_state"] = rs
    metrics["test_size"] = float(split_cfg.get("test_size", 0.2))
    metrics["stratify"] = bool(split_cfg.get("stratify", True))

    save_json(outdir, "metrics.json", metrics)

    # Fairness report (after)
    fair_cfg = cfg.get("fairness", {"enabled": False})
    if fair_cfg.get("enabled", False):
        sensitive_cols=cfg.get("fairness", {}).get("sensitive_cols", [])

        fair_df, fair_summary = compute_fairness_report(
            df_raw=df_raw.loc[X_test.index],
            y_true=y_test,
            y_pred=y_pred,
            sensitive_cols=cfg.get("fairness", {}).get("sensitive_cols", []),
            positive_label=int(cfg.get("fairness", {}).get("positive_label", 1)),
        )


        save_csv(outdir, "fairness_by_group.csv", fair_df)
        save_json(outdir, "fairness_summary.json", fair_summary)

    #Save model
    if cfg.get("outputs", {}).get("save_model", True):
        dump(model, outdir / "model.joblib")

    return metrics

