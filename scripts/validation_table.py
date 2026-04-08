from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fair_mitigator.io import load_config, load_csv, ensure_outdir
from fair_mitigator.pipelines.baseline import run_baseline
from fair_mitigator.pipelines.relabel import run_relabel
from fair_mitigator.pipelines.smote import run_smote
from fair_mitigator.pipelines.inprocess import run_inprocess
from fair_mitigator.metrics import evaluate
from fair_mitigator.models import make_model
from fair_mitigator.preprocessing import make_features
from fair_mitigator.fairness import compute_fairness_report
from fair_mitigator.mitigations.relabel import run_relabeling
from fair_mitigator.mitigations.latent import fit_transform_latent


SENSITIVE_COLS = ["sex", "race", "ethnicity"]

PIPELINES = {
    "baseline": {
        "config": "configs/baseline_test.yaml",
        "runner": "baseline",
    },
    "relabel": {
        "config": "configs/relabel_test.yaml",
        "runner": "relabel",
    },
    "smote": {
        "config": "configs/smote_test.yaml",
        "runner": "smote",
    },
    "latent": {
        "config": "configs/latent_test.yaml",
        "runner": "latent",
    },
    "inprocess_equalized_odds": {
        "config": "configs/inprocess_fair.yaml",
        "runner": "inprocess",
    },
}

DATASETS = {
    "no_disparity": {
        "file": "validation_data/no_disparity.csv",
        "issue_type": "control",
        "expected_best_method": "baseline",
        "notes": "Control dataset: baseline should already be fair.",
    },
    "class_imbalance_smote": {
        "file": "validation_data/class_imbalance_smote.csv",
        "issue_type": "class_imbalance",
        "expected_best_method": "smote",
        "notes": "Designed to test whether SMOTE helps when class imbalance is the main issue.",
    },
    "label_noise_relabel": {
        "file": "validation_data/label_noise_relabel.csv",
        "issue_type": "label_noise",
        "expected_best_method": "relabel",
        "notes": "Designed to test whether relabeling helps when labels near the boundary are noisy.",
    },
    "latent_disparity": {
        "file": "validation_data/latent_disparity.csv",
        "issue_type": "latent_structure",
        "expected_best_method": "latent",
        "notes": "Designed to test whether latent-variable mitigation helps with hidden structural bias.",
    },
    "eo_targeted": {
        "file": "validation_data/eo_targeted.csv",
        "issue_type": "equalized_odds_target",
        "expected_best_method": "inprocess_equalized_odds",
        "notes": "Designed to test equalized-odds style in-process mitigation.",
    },
}


def prepare_cfg(config_path: str) -> dict:
    cfg = load_config(REPO_ROOT / config_path)
    cfg = deepcopy(cfg)

    cfg.setdefault("fairness", {})
    cfg["fairness"]["enabled"] = True
    cfg["fairness"]["sensitive_cols"] = SENSITIVE_COLS
    cfg["fairness"]["positive_label"] = 1

    cfg.setdefault("outputs", {})
    cfg["outputs"]["save_model"] = False

    return cfg


def load_fairness_summary(outdir: Path) -> dict:
    summary_path = outdir / "fairness_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_latent_with_fairness(df: pd.DataFrame, cfg: dict, outdir: Path):
    rs = int(cfg.get("random_state", 42))

    X_all, y_all, df_raw = make_features(df, cfg["data"])

    split_cfg = cfg.get("split", {"test_size": 0.2, "stratify": True})
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=rs,
        stratify=y_all if split_cfg.get("stratify", True) else None,
    )

    lat_spec = cfg["data"].get("latent", {})
    if not lat_spec.get("enabled", True):
        raise ValueError("latent pipeline expects data.latent.enabled = true")

    Z_train_df, Z_test_df, _ = fit_transform_latent(
        df_raw.loc[X_train.index],
        df_raw.loc[X_test.index],
        lat_spec,
    )

    mode = lat_spec.get("mode", "original_plus_latent")
    if mode == "latent_only":
        X_train_use = Z_train_df
        X_test_use = Z_test_df
    else:
        X_train_use = X_train.join(Z_train_df, how="left")
        X_test_use = X_test.join(Z_test_df, how="left")

    rel_cfg = cfg.get("relabeling", {"enabled": False})
    if rel_cfg.get("enabled", False):
        rel_res = run_relabeling(X_train_use, y_train, rel_cfg, outdir=outdir)
        y_train_used = rel_res.y_train_cleaned
    else:
        y_train_used = y_train

    model = make_model(cfg["model"])
    model.fit(X_train_use, y_train_used)
    y_pred = model.predict(X_test_use)

    metrics = evaluate(y_test, y_pred)
    metrics["pipeline"] = "latent"
    metrics["latent_mode"] = mode
    metrics["latent_n_components"] = int(lat_spec.get("n_components", 2))

    _, fair_summary = compute_fairness_report(
        df_raw=df_raw.loc[X_test.index],
        y_true=y_test,
        y_pred=y_pred,
        sensitive_cols=cfg["fairness"]["sensitive_cols"],
        positive_label=int(cfg["fairness"].get("positive_label", 1)),
    )

    return metrics, fair_summary


def flatten_result(dataset_name: str, pipeline_name: str, metrics: dict, fair_summary: dict) -> dict:
    row = {
        "dataset": dataset_name,
        "pipeline": pipeline_name,
        "accuracy": round(float(metrics.get("accuracy", float("nan"))), 4),
    }

    disparities = []

    metric_map = [
        ("demographic_parity_difference", "dp"),
        ("equal_opportunity_difference", "eo"),
        ("equalized_odds_difference", "eod"),
    ]

    for sensitive_col in SENSITIVE_COLS:
        values = fair_summary.get(sensitive_col, {})
        for long_name, short_name in metric_map:
            value = values.get(long_name)
            col_name = f"{sensitive_col}_{short_name}"
            if value is None:
                row[col_name] = pd.NA
            else:
                value = float(value)
                row[col_name] = round(value, 4)
                disparities.append(abs(value))

    row["max_disparity"] = round(max(disparities), 4) if disparities else pd.NA
    return row


def make_markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy().astype(object)
    display_df = display_df.where(pd.notna(display_df), "NA")
    return display_df.to_markdown(index=False)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ensure_outdir(REPO_ROOT / "reports" / f"validation_tables_{stamp}")

    rows = []
    raw_results = {}

    for dataset_name, dataset_meta in DATASETS.items():
        dataset_path = REPO_ROOT / dataset_meta["file"]
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {dataset_path}\n"
                f"Run: python scripts/generate_validation_suite.py"
            )

        df = load_csv(dataset_path)
        raw_results[dataset_name] = {
            "metadata": dataset_meta,
            "pipelines": {},
        }

        for pipeline_name, meta in PIPELINES.items():
            cfg = prepare_cfg(meta["config"])
            outdir = ensure_outdir(report_dir / dataset_name / pipeline_name)

            if meta["runner"] == "baseline":
                metrics = run_baseline(df, cfg, outdir)
                fair_summary = load_fairness_summary(outdir)

            elif meta["runner"] == "relabel":
                metrics = run_relabel(df, cfg, outdir)
                fair_summary = load_fairness_summary(outdir)

            elif meta["runner"] == "smote":
                metrics = run_smote(df, cfg, outdir)
                fair_summary = load_fairness_summary(outdir)

            elif meta["runner"] == "inprocess":
                metrics = run_inprocess(df, cfg, outdir)
                fair_summary = load_fairness_summary(outdir)

            elif meta["runner"] == "latent":
                metrics, fair_summary = run_latent_with_fairness(df, cfg, outdir)

            else:
                raise ValueError(f"Unknown runner: {meta['runner']}")

            raw_results[dataset_name]["pipelines"][pipeline_name] = {
                "metrics": metrics,
                "fairness_summary": fair_summary,
            }

            row = flatten_result(
                dataset_name=dataset_name,
                pipeline_name=pipeline_name,
                metrics=metrics,
                fair_summary=fair_summary,
            )
            row["issue_type"] = dataset_meta["issue_type"]
            row["expected_best_method"] = dataset_meta["expected_best_method"]
            rows.append(row)

    table_df = pd.DataFrame(rows)

    baseline_lookup = (
        table_df[table_df["pipeline"] == "baseline"][["dataset", "max_disparity", "accuracy"]]
        .rename(
            columns={
                "max_disparity": "baseline_max_disparity",
                "accuracy": "baseline_accuracy",
            }
        )
    )

    table_df = table_df.merge(baseline_lookup, on="dataset", how="left")

    table_df["improved_vs_baseline"] = (
        table_df["max_disparity"] < table_df["baseline_max_disparity"]
    ).astype("boolean")
    table_df.loc[table_df["pipeline"] == "baseline", "improved_vs_baseline"] = pd.NA

    table_df["disparity_change_vs_baseline"] = (
        table_df["max_disparity"] - table_df["baseline_max_disparity"]
    )
    table_df.loc[table_df["pipeline"] == "baseline", "disparity_change_vs_baseline"] = pd.NA

    table_df["accuracy_change_vs_baseline"] = (
        table_df["accuracy"] - table_df["baseline_accuracy"]
    )
    table_df.loc[table_df["pipeline"] == "baseline", "accuracy_change_vs_baseline"] = pd.NA

    expected_rows = []
    for dataset_name, meta in DATASETS.items():
        expected_method = meta["expected_best_method"]
        sub = table_df[(table_df["dataset"] == dataset_name) & (table_df["pipeline"] == expected_method)]

        if sub.empty:
            expected_rows.append(
                {
                    "dataset": dataset_name,
                    "issue_type": meta["issue_type"],
                    "expected_best_method": expected_method,
                    "expected_method_found": False,
                    "expected_method_improved": pd.NA,
                    "expected_method_max_disparity": pd.NA,
                    "baseline_max_disparity": pd.NA,
                    "disparity_change_vs_baseline": pd.NA,
                    "accuracy_change_vs_baseline": pd.NA,
                    "notes": meta["notes"],
                }
            )
            continue

        row = sub.iloc[0]
        expected_rows.append(
            {
                "dataset": dataset_name,
                "issue_type": meta["issue_type"],
                "expected_best_method": expected_method,
                "expected_method_found": True,
                "expected_method_improved": row["improved_vs_baseline"],
                "expected_method_max_disparity": row["max_disparity"],
                "baseline_max_disparity": row["baseline_max_disparity"],
                "disparity_change_vs_baseline": row["disparity_change_vs_baseline"],
                "accuracy_change_vs_baseline": row["accuracy_change_vs_baseline"],
                "notes": meta["notes"],
            }
        )

    expected_df = pd.DataFrame(expected_rows)

    summary_rows = []
    for dataset_name, group in table_df.groupby("dataset", dropna=False):
        baseline_val = group.loc[group["pipeline"] == "baseline", "max_disparity"].iloc[0]
        improved_count = int(group["improved_vs_baseline"].fillna(False).sum())

        best_nonbaseline = group[group["pipeline"] != "baseline"].sort_values("max_disparity", ascending=True).iloc[0]

        summary_rows.append(
            {
                "dataset": dataset_name,
                "issue_type": group["issue_type"].iloc[0],
                "expected_best_method": group["expected_best_method"].iloc[0],
                "baseline_max_disparity": baseline_val,
                "pipelines_tested": int(group.shape[0]),
                "pipelines_better_than_baseline": improved_count,
                "best_nonbaseline_pipeline": best_nonbaseline["pipeline"],
                "best_nonbaseline_max_disparity": best_nonbaseline["max_disparity"],
                "best_nonbaseline_delta_vs_baseline": best_nonbaseline["disparity_change_vs_baseline"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    table_df = table_df[
        [
            "dataset",
            "issue_type",
            "expected_best_method",
            "pipeline",
            "accuracy",
            "sex_dp",
            "sex_eo",
            "sex_eod",
            "race_dp",
            "race_eo",
            "race_eod",
            "ethnicity_dp",
            "ethnicity_eo",
            "ethnicity_eod",
            "max_disparity",
            "baseline_max_disparity",
            "improved_vs_baseline",
            "disparity_change_vs_baseline",
            "accuracy_change_vs_baseline",
        ]
    ]

    table_df.to_csv(report_dir / "validation_table.csv", index=False)
    summary_df.to_csv(report_dir / "validation_summary.csv", index=False)
    expected_df.to_csv(report_dir / "validation_expected_method_summary.csv", index=False)

    (report_dir / "validation_table.md").write_text(make_markdown_table(table_df), encoding="utf-8")
    (report_dir / "validation_summary.md").write_text(make_markdown_table(summary_df), encoding="utf-8")
    (report_dir / "validation_expected_method_summary.md").write_text(
        make_markdown_table(expected_df),
        encoding="utf-8",
    )
    (report_dir / "validation_raw_results.json").write_text(
        json.dumps(raw_results, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION TABLE\n")
    print(make_markdown_table(table_df))

    print("\nSUMMARY\n")
    print(make_markdown_table(summary_df))

    print("\nEXPECTED METHOD SUMMARY\n")
    print(make_markdown_table(expected_df))

    print(f"\nSaved results to: {report_dir}")


if __name__ == "__main__":
    main()
