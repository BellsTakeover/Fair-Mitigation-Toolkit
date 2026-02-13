from __future__ import annotations

from joblib import dump
from sklearn.model_selection import train_test_split

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

from ..io import save_csv, save_json
from ..metrics import evaluate, cross_validate
from ..models import make_model
from ..preprocessing import make_features
from ..fairness import compute_fairness_report
from ..mitigations.relabel import run_relabeling


def run_inprocess(df, cfg, outdir):
    if cfg is None:
        raise ValueError("Config file loaded as empty. Check your --config path.")

    rs = int(cfg.get("random_state", 42))

    #build numeric X plus raw frame (raw is used for sensitive columns)
    X_all, y_all, df_raw = make_features(df, cfg["data"])

    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y_all if split_cfg.get("stratify", True) else None,
    )

    # Optional relabeling is still "pre-processing", but it pairs nicely with in-processing runs.
    rel_cfg = cfg.get("relabeling", {"enabled": False})
    if rel_cfg.get("enabled", False):
        rel_res = run_relabeling(X_train, y_train, rel_cfg, outdir=outdir)
        y_train_used = rel_res.y_train_cleaned
    else:
        rel_res = None
        y_train_used = y_train

    in_cfg = cfg.get("inprocessing", {})
    mode = in_cfg.get("mode", "rf_regularized")

    #RF-only model
    base_est = make_model(cfg["model"])

    fair_cfg = cfg.get("fairness", {"enabled": False})
    sens_cols = fair_cfg.get("sensitive_cols", [])

    #If we're doing fairness-aware in-processing, we need a single raw sensitive column for training.
    sens_col = in_cfg.get("sensitive_col", sens_cols[0] if sens_cols else None)

    if mode == "rf_regularized":
        base_est.fit(X_train, y_train_used)
        y_pred = base_est.predict(X_test)
        model_obj = base_est
        model_name = "rf_regularized"

    elif mode == "expgrad_equalized_odds":
        if not sens_col:
            raise ValueError("Set inprocessing.sensitive_col (must be a raw CSV column).")
        if sens_col not in df_raw.columns:
            raise ValueError(f"inprocessing.sensitive_col '{sens_col}' not found in the CSV.")

        mitigator = ExponentiatedGradient(
            estimator=base_est,
            constraints=EqualizedOdds(),
        )
        mitigator.fit(
            X_train,
            y_train_used,
            sensitive_features=df_raw.loc[X_train.index, sens_col],
        )
        y_pred = mitigator.predict(X_test)
        model_obj = mitigator
        model_name = "expgrad_equalized_odds_rf"

    else:
        raise ValueError(f"Unknown inprocessing mode: {mode}")

    # Metrics
    results = evaluate(y_test, y_pred)
    results["pipeline"] = "inprocess"
    results["inprocessing_mode"] = mode
    results["model_name"] = model_name
    results["random_state"] = rs
    results["test_size"] = float(split_cfg.get("test_size", 0.2))

    if rel_cfg.get("enabled", False) and rel_res is not None:
        results["relabeling_enabled"] = True
        results["relabel_threshold"] = float(rel_cfg.get("confidence_threshold", 0.9))
        results["num_suspicious"] = int(rel_res.suspicious_mask.sum())
    else:
        results["relabeling_enabled"] = False

    save_json(outdir, "metrics.json", results)

    # Fairness report (optional)
    if fair_cfg.get("enabled", False):
        if not sens_cols:
            raise ValueError("fairness.enabled is true but fairness.sensitive_cols is empty.")

        fair_df, fair_summary = compute_fairness_report(
            df_raw=df_raw.loc[X_test.index],
            y_true=y_test,
            y_pred=y_pred,
            sensitive_cols=sens_cols,
            positive_label=int(fair_cfg.get("positive_label", 1)),
        )
        save_csv(outdir, "fairness_by_group.csv", fair_df)
        save_json(outdir, "fairness_summary.json", fair_summary)

    # Cross-validation (kept simple: CV only for rf_regularized mode)
    cv_cfg = cfg.get("cross_validation", {"enabled": False})
    if cv_cfg.get("enabled", False) and mode == "rf_regularized":
        est_cv = make_model(cfg["model"])
        results.update(cross_validate(est_cv, X_all, y_all, cv_cfg, random_state=rs))
        save_json(outdir, "metrics.json", results)

    # Save model
    if cfg.get("outputs", {}).get("save_model", True):
        dump(model_obj, outdir / "model.joblib")

    return results
