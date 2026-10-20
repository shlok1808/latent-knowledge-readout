"""Private W&B logging for the multistep-gap runs. No-op when WANDB_API_KEY is unset or
wandb is not installed, so every pipeline still runs without it.

Entity: `agentic-interp`, a team whose only member is shlok05 (there is no personal
entity; `mo-aayuan-khan-` has other members, so never use it). Project forced private.

    from mstep import wb
    run = wb.start("ms-tc-1p7b", job="probes", config={...})
    wb.table(run, "phase2_cells", df); wb.image(run, "fig", path); wb.artifact(run, "results", dir)
    wb.finish(run)
"""
import os
from pathlib import Path

ENTITY = os.environ.get("WANDB_ENTITY", "agentic-interp")  # team whose only member is shlok05 (checked 2026-09-24)
PROJECT = os.environ.get("WANDB_PROJECT", "agent-failure-probes-multistep")


def _wandb():
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def start(name, job, config=None, group=None):
    wandb = _wandb()
    if wandb is None:
        return None
    return wandb.init(entity=ENTITY, project=PROJECT, name=f"{name}:{job}", group=group or name, job_type=job,
                      config=config or {}, reinit="create_new", settings=wandb.Settings(silent=True))


def log(run, data, step=None):
    if run is not None:
        run.log(data, step=step)


def table(run, key, df):
    if run is not None and df is not None and len(df):
        import wandb
        run.log({key: wandb.Table(dataframe=df.drop(columns=[c for c in ("text", "surface") if c in df.columns]))})


def image(run, key, path):
    if run is not None and Path(path).exists():
        import wandb
        run.log({key: wandb.Image(str(path))})


def artifact(run, name, path, kind="results"):
    if run is not None and Path(path).exists():
        import wandb
        art = wandb.Artifact(name.replace("/", "-").replace(">=", "ge"), type=kind)
        (art.add_dir if Path(path).is_dir() else art.add_file)(str(path))
        run.log_artifact(art)


def finish(run):
    if run is not None:
        run.finish()


def backfill_results(results_dir, run_name):
    """Log an existing results directory: config, every CSV as a table, every PNG, all as an artifact."""
    import json
    import pandas as pd
    d = Path(results_dir)
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    run = start(run_name, "results", config=cfg)
    if run is None:
        return None
    for csv in sorted(d.glob("*.csv")):
        table(run, csv.stem, pd.read_csv(csv))
    for png in sorted(d.glob("*.png")):
        image(run, f"fig/{png.stem}", png)
    artifact(run, run_name, d)
    url = run.url
    finish(run)
    return url


if __name__ == "__main__":
    import sys
    print(backfill_results(sys.argv[1], sys.argv[2]))
