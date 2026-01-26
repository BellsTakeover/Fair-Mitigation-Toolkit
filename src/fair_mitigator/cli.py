import argparse
from datetime import datetime
from pathlib import Path

from fair_mitigator.io import load_config, load_csv, ensure_outdir
from fair_mitigator.pipelines.baseline import run_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.data)

    outdir = ensure_outdir(Path("reports") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if args.pipeline == "baseline":
        run_baseline(df, cfg, outdir)
    else:
        raise ValueError("Unknown pipeline")

    print("Done.")

if __name__ == "__main__":
    main()
