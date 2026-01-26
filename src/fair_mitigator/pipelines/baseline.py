from sklearn.model_selection import train_test_split
from joblib import dump

from fair_mitigator.preprocessing import make_features
from fair_mitigator.models import make_model, run_grid_search
from fair_mitigator.fairness import compute_fairness_report
from fair_mitigator.io import save_csv

def run_baseline(df, cfg, outdir):
    X, y = make_features(df, cfg["data"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["random_state"],
        stratify=y if cfg["split"]["stratify"] else None
    )

    model = make_model(cfg["model"])
    model, _ = run_grid_search(model, cfg["grid_search"], X_train, y_train)

    y_pred = model.predict(X_test)

    if cfg["fairness"]["enabled"]:
        fair_df = compute_fairness_report(
            df.loc[X_test.index],
            y_test,
            y_pred,
            cfg["fairness"]["sensitive_cols"]
        )
        save_csv(outdir, "fairness_by_group.csv", fair_df)

    dump(model, outdir / "model.joblib")
