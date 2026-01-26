#loads CSV, and YAML
#also saves outputs
from pathlib import Path
import json
import pandas as pd
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_csv(path):
    return pd.read_csv(path)

def ensure_outdir(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def save_used_config(outdir, cfg):
    with open(outdir / "params_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

def save_json(outdir, name, obj):
    with open(outdir / name, "w") as f:
        json.dump(obj, f, indent=2)

def save_text(outdir, name, text):
    with open(outdir / name, "w") as f:
        f.write(text)

def save_csv(outdir, name, df):
    df.to_csv(outdir / name, index=False)
