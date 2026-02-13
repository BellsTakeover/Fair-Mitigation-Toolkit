#GOAL: pipline routing 
import argparse
from datetime import datetime
from pathlib import Path

#runs all the pipelines
from fair_mitigator.io import load_config, load_csv, ensure_outdir
from fair_mitigator.pipelines.baseline import run_baseline
from fair_mitigator.pipelines.relabel import run_relabel
from fair_mitigator.pipelines.smote import run_smote
from fair_mitigator.pipelines.latent import run_latent
from fair_mitigator.pipelines.inprocess import run_inprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["baseline", "relabel", "smote", "latent", "inprocess"],
        help="Which bias-mitigation pipeline to run"
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.data)

    outdir = ensure_outdir(Path("reports") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if args.pipeline == "baseline":
        run_baseline(df, cfg, outdir)
    elif args.pipeline == "relabel":
        run_relabel(df, cfg, outdir)
    elif args.pipeline == "smote":
        run_smote(df, cfg, outdir)
    elif args.pipeline == "latent":
        run_latent(df, cfg, outdir)
    elif args.pipeline == "inprocess":
        run_inprocess(df, cfg, outdir)
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")


    print("Done.")

if __name__ == "__main__":
    main()
