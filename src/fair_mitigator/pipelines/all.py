from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..io import ensure_outdir, save_json
from .baseline import run_baseline
from .smote import run_smote
from .latent import run_latent

# We treat "relabel" as baseline + relabeling enabled (same model path, same outputs)
# so users don't need a separate mental model for it.
# If you still have a dedicated pipelines/relabel.py, this keeps things consistent.


def _child_outdir(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_all(df, cfg: dict, outdir: Path):
    """
    Runs every mitigation technique in sequence.

    Layout:
      reports/run_xxx_all/
        baseline/
        relabel/
        smote/
        latent/
        rf_regularized/

    Notes:
      - Each sub-pipeline writes its own metrics.json + artifacts into its folder.
      - If a sub-config is missing, we warn and skip it instead of failing the whole run.
    """

    results_summary = {
        "pipeline": "all",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": {},
        "skipped": [],
    }

    # 1) Baseline
    try:
        subcfg = cfg.get("baseline")
        if subcfg is None:
            results_summary["skipped"].append("baseline (missing cfg.baseline)")
        else:
            d = _child_outdir(outdir, "baseline")
            res = run_baseline(df, subcfg, d)
            results_summary["runs"]["baseline"] = res
    except Exception as e:
        results_summary["runs"]["baseline_error"] = str(e)

    # 2) Relabel (baseline pipeline with relabeling enabled)
    try:
        subcfg = cfg.get("relabel")
        if subcfg is None:
            results_summary["skipped"].append("relabel (missing cfg.relabel)")
        else:
            d = _child_outdir(outdir, "relabel")
            res = run_baseline(df, subcfg, d)
            results_summary["runs"]["relabel"] = res
    except Exception as e:
        results_summary["runs"]["relabel_error"] = str(e)

    # 3) SMOTE
    try:
        subcfg = cfg.get("smote")
        if subcfg is None:
            results_summary["skipped"].append("smote (missing cfg.smote)")
        else:
            d = _child_outdir(outdir, "smote")
            res = run_smote(df, subcfg, d)
            results_summary["runs"]["smote"] = res
    except Exception as e:
        results_summary["runs"]["smote_error"] = str(e)

    # 4) Latent variables
    try:
        subcfg = cfg.get("latent")
        if subcfg is None:
            results_summary["skipped"].append("latent (missing cfg.latent)")
        else:
            d = _child_outdir(outdir, "latent")
            res = run_latent(df, subcfg, d)
            results_summary["runs"]["latent"] = res
    except Exception as e:
        results_summary["runs"]["latent_error"] = str(e)

    # 5) RF "regularized" settings (still RF, just tighter constraints)
    try:
        subcfg = cfg.get("rf_regularized")
        if subcfg is None:
            results_summary["skipped"].append("rf_regularized (missing cfg.rf_regularized)")
        else:
            d = _child_outdir(outdir, "rf_regularized")
            res = run_baseline(df, subcfg, d)
            results_summary["runs"]["rf_regularized"] = res
    except Exception as e:
        results_summary["runs"]["rf_regularized_error"] = str(e)

    save_json(outdir, "all_summary.json", results_summary)
    return results_summary
