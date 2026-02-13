from __future__ import annotations

from joblib import dump
from sklearn.model_selection import train_test_split

from ..io import save_json, save_csv
from ..metrics import evaluate, cross_validate
from ..models import make_model
from ..preprocessing import make_features
from ..fairness import compute_fairness_report
from ..mitigations.relabel import run_relabeling


def run_relabel(df, cfg, outdir):
    rs = int(cfg.get("random_state", 42))

    X_all, y_all, df_raw = make_features(df, cfg["data"])

    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y_all if split_cfg.get("stratify", True) else None,
    )

    rel_cfg = cfg.get("relabeling", {"enabled": True})
    if not rel_cfg.get("enabled", True):
        raise ValueError("relabel pipeline expects relabeling.enabled = true")

    rel_res = run_relabeling(X_train, y_train, rel_cfg, outdir=outdir)

    model = make_model(cfg["model"])
    model.fit(X_train, rel_res.y_train_cleaned)
    y_pred = model.predict(X_test)

    results = evaluate(y_test, y_pred)
    results["pipeline"] = "relabel"
    results["num_suspicious"] = int(rel_res.suspicious_mask.sum())
    results["relabel_threshold"] = float(rel_cfg.get("confidence_threshold", 0.9))
    save_json(outdir, "metrics.json", results)

    fair_cfg = cfg.get("fairness", {"enabled": False})
    if fair_cfg.get("enabled", False):
        sens_cols = fair_cfg.get("sensitive_cols", [])
        fair_df, fair_summary = compute_fairness_report(
            df_raw=df_raw.loc[X_test.index],
            y_true=y_test,
            y_pred=y_pred,
            sensitive_cols=sens_cols,
            positive_label=int(fair_cfg.get("positive_label", 1)),
        )
        save_csv(outdir, "fairness_by_group.csv", fair_df)
        save_json(outdir, "fairness_summary.json", fair_summary)

    cv_cfg = cfg.get("cross_validation", {"enabled": False})
    if cv_cfg.get("enabled", False):
        est_cv = make_model(cfg["model"])
        results.update(cross_validate(est_cv, X_all, y_all, cv_cfg, random_state=rs))
        save_json(outdir, "metrics.json", results)

    if cfg.get("outputs", {}).get("save_model", True):
        dump(model, outdir / "model.joblib")

    return results



