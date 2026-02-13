from __future__ import annotations

from joblib import dump
from sklearn.model_selection import train_test_split

from ..io import save_json, save_csv
from ..metrics import evaluate, cross_validate
from ..models import make_model
from ..preprocessing import make_features
from ..fairness import compute_fairness_report
from ..mitigations.relabel import run_relabeling
from ..mitigations.latent import fit_transform_latent


def run_latent(df, cfg, outdir):
    rs = int(cfg.get("random_state", 42))

    #build the usual numeric matrix (works for any dataset)
    X_all, y_all, df_raw = make_features(df, cfg["data"])

    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y_all if split_cfg.get("stratify", True) else None,
    )

    lat_spec = cfg["data"].get("latent", {})
    if not lat_spec.get("enabled", True):
        raise ValueError("latent pipeline expects data.latent.enabled = true")

    #Fit PCA on raw train, transform raw train/test --> in order to work on any dataset
    Z_train_df, Z_test_df, latent_pipe = fit_transform_latent(
        df_raw.loc[X_train.index],
        df_raw.loc[X_test.index],
        lat_spec
    )

    mode = lat_spec.get("mode", "original_plus_latent")
    if mode == "latent_only":
        X_train_use = Z_train_df
        X_test_use = Z_test_df

        #Provides proof that latent variables were properly added 
        print("[DEBUG] X_train_use shape:", X_train_use.shape)
        print( "[DEBUG] latent columns:",
        [c for c in X_train_use.columns if str(c).startswith(str(lat_spec.get("prefix", "Latent")))])

    else:
        #original_plus_latent
        X_train_use = X_train.join(Z_train_df, how="left")
        X_test_use = X_test.join(Z_test_df, how="left")

    #Provides proof that latent variables were properly added 
    print("[DEBUG] X_train_use shape:", X_train_use.shape)
    print( "[DEBUG] latent columns:",
    [c for c in X_train_use.columns if str(c).startswith(str(lat_spec.get("prefix", "Latent")))])
    
    #Optional relabeling (after latent features are added) --> if further pre-processing bias mitigation
    rel_cfg = cfg.get("relabeling", {"enabled": False})
    if rel_cfg.get("enabled", False):
        rel_res = run_relabeling(X_train_use, y_train, rel_cfg, outdir=outdir)
        y_train_used = rel_res.y_train_cleaned
    else:
        rel_res = None
        y_train_used = y_train

    model = make_model(cfg["model"])
    model.fit(X_train_use, y_train_used)
    y_pred = model.predict(X_test_use)

    results = evaluate(y_test, y_pred)
    results["pipeline"] = "latent"
    results["latent_mode"] = mode
    results["latent_n_components"] = int(lat_spec.get("n_components", 2))

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
        #CV with latent is possible but it needs a fold-wise fit of PCA
        #Keeping it off by default avoids silent leakage
        pass

    if cfg.get("outputs", {}).get("save_model", True):
        dump(model, outdir / "model.joblib")

    return results
