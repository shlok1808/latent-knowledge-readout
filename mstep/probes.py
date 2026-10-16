"""Phase 2 (probe strategies) and Phase 3 (stage generalisation).

Everything uses ONE task-grouped 5-fold assignment per run (balanced by depth),
shared by all cells and all stages, so no task is ever in both train and test,
and transfer matrices compare probes that saw exactly the same tasks.

Phase 2 cell = (label, input S1/S2, round t, layer, probe logreg/dom) on the rows
alive at round t. Baselines per (label, t): TF-IDF text, surface counts, both.

Phase 3 = train on stage A, test on stage B, for round stages (early/mid/late)
and milestone stages (k-th non-terminal craft), under two populations:
  naive  : every row alive in the stage (Ruan's convention)
  cohort : only episodes present in EVERY compared stage (same episodes, same
           base rate, only time differs) -- the primary analysis (A-12).
Plus per-test-round curves (x = round t) for strategies {train early, mid, late,
all-rounds pooled, same-round} and a pooled TF-IDF baseline.

  python -m mstep.probes --run runs/ms-tc-1p7b --out results/multistep/textcraft_qwen3-1.7b [--phase 2|3|all]
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mstep.metrics import summarize

LABELS = ["final_bad", "mc_bad", "valid_bad", "noprogress_bad"]
PHASE2_T = [1, 2, 3, 5, 8]
STAGES = ["early", "mid", "late"]
LASTSTEP = {"early": 2, "mid": 5, "late": 10}  # representative (last) round of each range
SEED = 42
FOLDS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data access

def layer_files(run, site, n_layers):
    """Split acts/{site}.npy [N, L+1, d] into contiguous per-layer files once."""
    src = run / "acts" / f"{site}.npy"
    out = run / "acts" / "by_layer"
    out.mkdir(exist_ok=True)
    paths = [out / f"{site}_L{l:02d}.npy" for l in range(n_layers + 1)]
    if all(p.exists() for p in paths):
        return paths
    acts = np.load(src, mmap_mode="r")
    step = 4096
    arrays = [np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(acts.shape[0], acts.shape[2])) for p in paths]
    for i in range(0, acts.shape[0], step):
        block = np.asarray(acts[i:i + step])
        for l, arr in enumerate(arrays):
            arr[i:i + step] = block[:, l]
    for arr in arrays:
        arr.flush()
    return paths


def load_layer(path, rows=None):
    x = np.load(path, mmap_mode="r")
    return np.asarray(x if rows is None else x[rows], dtype=np.float32)


def mean_pool(x_all, df):
    """S2: mean of rounds 1..t of the same episode, for every row (rows are episode-major, round-ordered)."""
    out = np.empty_like(x_all)
    for _, ix in df.groupby("episode_id", sort=False).indices.items():
        ix = np.sort(ix)
        c = np.cumsum(x_all[ix], axis=0)
        out[ix] = c / np.arange(1, len(ix) + 1)[:, None]
    return out


def assign_folds(df):
    """Task -> fold, balanced within each depth, fixed seed."""
    rng = np.random.default_rng(SEED)
    fold = {}
    for _, tasks in df.groupby("depth")["task_id"].unique().items():
        tasks = sorted(tasks)
        rng.shuffle(tasks)
        for i, t in enumerate(tasks):
            fold[t] = i % FOLDS
    return df["task_id"].map(fold).to_numpy()


# ---------------------------------------------------------------- probes

GPU_PROBES = os.environ.get("MSTEP_GPU_PROBES") == "1"


def _torch_logreg(x, y, C=1.0):
    """Same objective as sklearn LogisticRegression(C, lbfgs) on StandardScaler features:
    sum log-loss + ||w||^2 / (2C), intercept unpenalised; solved with L-BFGS on the GPU."""
    import torch
    X = torch.as_tensor(x, device="cuda", dtype=torch.float32)
    mu = X.mean(0)
    sd = X.std(0, unbiased=False)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    yt = torch.as_tensor(y, device="cuda", dtype=torch.float32)
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=1000, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, yt, reduction="sum") + (w @ w) / (2 * C)
        loss.backward()
        return loss
    opt.step(closure)
    return ("torch", mu, sd, w.detach(), b.detach())


def fit_probe(kind, x, y):
    if len(np.unique(y)) < 2:
        return None
    if kind == "logreg" and GPU_PROBES:
        return _torch_logreg(x, y)
    if kind == "logreg":
        scaler = StandardScaler().fit(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(scaler.transform(x), y)
        return ("logreg", scaler, clf)
    mu = x.mean(0)
    w = x[y == 1].mean(0) - x[y == 0].mean(0)
    return ("dom", mu, w / (np.linalg.norm(w) + 1e-8))


def score_probe(probe, x):
    if probe is None:
        return np.full(len(x), np.nan)
    if probe[0] == "logreg":
        return probe[2].predict_proba(probe[1].transform(x))[:, 1]
    if probe[0] == "torch":
        import torch
        _, mu, sd, w, b = probe
        X = torch.as_tensor(x, device="cuda", dtype=torch.float32)
        return torch.sigmoid(((X - mu) / sd) @ w + b).cpu().numpy()
    return (x - probe[1]) @ probe[2]


def oof(kind, x, y, folds, train_mask, test_mask):
    """Out-of-fold scores on test_mask rows, probes fit on train_mask rows of the other folds."""
    s = np.full(len(y), np.nan)
    for f in range(FOLDS):
        tr = train_mask & (folds != f) & np.isfinite(y)
        te = test_mask & (folds == f)
        if not te.any() or tr.sum() < 10:
            continue
        s[te] = score_probe(fit_probe(kind, x[tr], y[tr].astype(int)), x[te])
    return s


def episode_weights(df, mask):
    """1 / (#rows the episode contributes to this evaluation set)."""
    w = np.zeros(len(df))
    counts = df.loc[mask, "episode_id"].map(df.loc[mask, "episode_id"].value_counts())
    w[mask] = 1.0 / counts.to_numpy()
    return w


# ---------------------------------------------------------------- baselines

def text_scores(df, y, folds, train_mask, test_mask, kind):
    s = np.full(len(df), np.nan)
    surf = np.stack(df["surface"].to_numpy()).astype(float)
    for f in range(FOLDS):
        tr = np.where(train_mask & (folds != f) & np.isfinite(y))[0]
        te = np.where(test_mask & (folds == f))[0]
        if not len(te) or len(np.unique(y[tr])) < 2:
            continue
        parts_tr, parts_te = [], []
        if kind in ("tfidf", "tfidf_surface"):
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000, sublinear_tf=True)
            parts_tr.append(vec.fit_transform(df["text"].iloc[tr]))
            parts_te.append(vec.transform(df["text"].iloc[te]))
        if kind in ("surface", "tfidf_surface"):
            sc = StandardScaler().fit(surf[tr])
            parts_tr.append(csr_matrix(sc.transform(surf[tr])))
            parts_te.append(csr_matrix(sc.transform(surf[te])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(hstack(parts_tr).tocsr(), y[tr].astype(int))
        s[te] = clf.predict_proba(hstack(parts_te).tocsr())[:, 1]
    return s


# ---------------------------------------------------------------- phase 2

def phase2_layer(path, df, folds, layer):
    x1 = load_layer(path)
    x2 = mean_pool(x1, df)
    groups = df["task_id"].to_numpy()
    out = []
    for label in LABELS:
        y = df[label].to_numpy(dtype=float)
        for t in PHASE2_T:
            m = (df["round"].to_numpy() == t) & np.isfinite(y)
            if m.sum() < 20:
                continue
            for inp, x in (("S1", x1), ("S2", x2)):
                for kind in ("logreg", "dom"):
                    s = oof(kind, x, y, folds, m, m)
                    r = summarize(y[m], s[m], groups[m], prob=(kind == "logreg"))
                    out.append({"label": label, "input": inp, "t": t, "layer": layer, "probe": kind, **r})
    return out


def phase2(run, df, folds, paths, out_dir, n_jobs):
    log(f"phase 2: {len(paths)} layers x {len(LABELS)} labels x {len(PHASE2_T)} rounds x 2 inputs x 2 probes")
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(phase2_layer)(p, df, folds, l) for l, p in enumerate(paths))
    cells = pd.DataFrame([r for part in res for r in part])
    cells.to_csv(out_dir / "phase2_cells.csv", index=False)
    groups = df["task_id"].to_numpy()

    def base(label, t, kind):
        y = df[label].to_numpy(dtype=float)
        m = (df["round"].to_numpy() == t) & np.isfinite(y)
        if m.sum() < 20:
            return None
        s = text_scores(df, y, folds, m, m, kind)
        return {"label": label, "t": t, "baseline": kind, **summarize(y[m], s[m], groups[m], prob=True)}
    jobs = [(l, t, k) for l in LABELS for t in PHASE2_T for k in ("tfidf", "surface", "tfidf_surface")]
    res = Parallel(n_jobs=n_jobs)(delayed(base)(*j) for j in jobs)
    bases = pd.DataFrame([r for r in res if r])
    bases.to_csv(out_dir / "phase2_baselines.csv", index=False)
    log("phase 2 done")
    return cells, bases


def pick_layers(cells, n_layers, ruan_layer):
    """A-14: best 3 layers by mean within-task AUC across t (FINAL, S1, logreg), plus Ruan's layer."""
    c = cells[(cells.label == "final_bad") & (cells.input == "S1") & (cells.probe == "logreg") & ~cells.underpowered]
    if c.empty:
        c = cells[(cells.label == "final_bad") & (cells.input == "S1") & (cells.probe == "logreg")]
    rank = c.groupby("layer")["within_auc"].mean().sort_values(ascending=False)
    best = [int(l) for l in rank.index[:3]]
    layers = sorted(set(best + [ruan_layer]))
    return layers, rank


# ---------------------------------------------------------------- phase 3

def stage_masks(df, scheme, subset):
    """{stage_name: row mask} under 'naive' or 'cohort' population for a scheme."""
    r = df["round"].to_numpy()
    if scheme == "round":
        masks = {"early": (r >= 1) & (r <= 2), "mid": (r >= 3) & (r <= 5), "late": r >= 6}
        need = 6  # present in all three stages
        members = df["n_rounds"].to_numpy() >= need
    elif scheme == "phase":
        ph = df["phase"].to_numpy()
        masks = {k: ph == k for k in ("identify", "lookup", "act", "close")}
        have = df.groupby("episode_id")["phase"].transform(lambda x: set(x) >= {"identify", "lookup", "act", "close"})
        members = have.to_numpy(dtype=bool)
    elif scheme == "laststep":
        # Shlok 2026-09-24: each range is represented by its LAST step, not an average over it.
        # early 1-2 -> round 2, mid 3-5 -> round 5, late 6-10 -> round 10 (late capped at 10:
        # an open-ended range has no last step, and an episode's final round leaks the outcome).
        masks = {f"early@{LASTSTEP['early']}": r == LASTSTEP["early"], f"mid@{LASTSTEP['mid']}": r == LASTSTEP["mid"],
                 f"late@{LASTSTEP['late']}": r == LASTSTEP["late"]}
        members = df["n_rounds"].to_numpy() >= LASTSTEP["late"]
    else:  # milestone k = the round of the k-th non-terminal craft
        ms = pd.to_numeric(df["milestone"], errors="coerce").to_numpy()
        masks = {f"m{k}": np.nan_to_num(ms, nan=0) == k for k in (1, 2, 3)}
        reached = df.groupby("episode_id")["milestone"].transform("max").fillna(0).to_numpy()
        members = reached >= 3
        if subset == "cohort":
            # fall back to 2 milestones when too few episodes reach 3 (logged in the output)
            if df.loc[members, "task_id"].nunique() < 15:
                masks = {k: v for k, v in masks.items() if k != "m3"}
                members = reached >= 2
    if subset == "cohort":
        masks = {k: v & members for k, v in masks.items()}
    return masks


def phase3_layer(path, df, folds, layer, scheme, population, kinds=("logreg", "dom")):
    x = load_layer(path)
    groups = df["task_id"].to_numpy()
    masks = stage_masks(df, scheme, population)
    allm = np.zeros(len(df), bool)
    for m in masks.values():
        allm |= m
    out = []
    for label in LABELS:
        y = df[label].to_numpy(dtype=float)
        ok = np.isfinite(y)
        trains = dict(masks, pooled=allm)
        for kind in kinds:
            for a, ma in trains.items():
                for b, mb in masks.items():
                    test = mb & ok
                    if test.sum() < 20:
                        continue
                    s = oof(kind, x, y, folds, ma & ok, test)
                    w = episode_weights(df, test)
                    r = summarize(y[test], s[test], groups[test], w[test], prob=(kind == "logreg"))
                    out.append({"scheme": scheme, "population": population, "label": label, "layer": layer,
                                "probe": kind, "train": a, "test": b, **r})
    return out


def round_curve_layer(path, df, folds, layer, population, t_max=10, kind="logreg"):
    """x = test round t; strategies trained on early/mid/late/pooled/same-round."""
    x = load_layer(path)
    groups = df["task_id"].to_numpy()
    r = df["round"].to_numpy()
    members = np.ones(len(df), bool) if population == "naive" else df["n_rounds"].to_numpy() >= t_max
    strategies = {"train_early": (r <= 2), "train_mid": (r >= 3) & (r <= 5), "train_late": r >= 6,
                  "train_pooled": r >= 1}
    out = []
    for label in LABELS:
        y = df[label].to_numpy(dtype=float)
        ok = np.isfinite(y) & members
        scores = {k: oof(kind, x, y, folds, m & ok, ok) for k, m in strategies.items()}
        for t in range(1, t_max + 1):
            test = ok & (r == t)
            if test.sum() < 20:
                continue
            same = oof(kind, x, y, folds, test, test)
            for k, s in list(scores.items()) + [("train_same_round", same)]:
                res = summarize(y[test], s[test], groups[test], prob=(kind == "logreg"))
                out.append({"population": population, "label": label, "layer": layer, "probe": kind,
                            "strategy": k, "t": t, **res})
    return out


def text_curve(df, folds, population, t_max=10):
    groups = df["task_id"].to_numpy()
    r = df["round"].to_numpy()
    members = np.ones(len(df), bool) if population == "naive" else df["n_rounds"].to_numpy() >= t_max
    out = []
    for label in LABELS:
        y = df[label].to_numpy(dtype=float)
        ok = np.isfinite(y) & members & (r <= t_max)
        s = text_scores(df, y, folds, ok, ok, "tfidf_surface")
        for t in range(1, t_max + 1):
            test = ok & (r == t)
            if test.sum() < 20:
                continue
            out.append({"population": population, "label": label, "layer": -1, "probe": "tfidf_surface",
                        "strategy": "text_pooled", "t": t, **summarize(y[test], s[test], groups[test], prob=True)})
    return out


def phase3(run, df, folds, paths, layers, out_dir, n_jobs, t_max):
    jobs = []
    for layer in layers:
        for population in ("naive", "cohort"):
            jobs.append(delayed(phase3_layer)(paths[layer], df, folds, layer, "round", population))
            jobs.append(delayed(phase3_layer)(paths[layer], df, folds, layer, "laststep", population))
            if "phase" in df.columns and df["phase"].notna().any():
                jobs.append(delayed(phase3_layer)(paths[layer], df, folds, layer, "phase", population))
            if (df["milestone"].notna()).any():
                jobs.append(delayed(phase3_layer)(paths[layer], df, folds, layer, "milestone", population))
            jobs.append(delayed(round_curve_layer)(paths[layer], df, folds, layer, population, t_max))
    for population in ("naive", "cohort"):
        jobs.append(delayed(text_curve)(df, folds, population, t_max))
    log(f"phase 3: {len(jobs)} jobs on layers {layers}")
    res = Parallel(n_jobs=n_jobs)(jobs)
    rows = [r for part in res for r in part]
    mat = pd.DataFrame([r for r in rows if "train" in r])
    cur = pd.DataFrame([r for r in rows if "strategy" in r])
    mat.to_csv(out_dir / "phase3_transfer.csv", index=False)
    cur.to_csv(out_dir / "phase3_round_curves.csv", index=False)
    log("phase 3 done")
    return mat, cur


def population_table(df, t_max):
    """Who is alive at each round: the size of the confound the cohort control removes."""
    ep = df.groupby("episode_id").first()
    rows = []
    for t in range(1, 21):
        alive = ep[ep.n_rounds >= t]
        if alive.empty:
            break
        mixed = alive.groupby("task_id")["final_bad"].agg(lambda s: s.nunique() == 2).sum()
        rows.append({"t": t, "alive": len(alive), "fail_share": alive.final_bad.mean(), "mixed_tasks": int(mixed),
                     "cohort_member_share": float((alive.n_rounds >= t_max).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=["2", "3", "3L", "3X", "all"],
                    help="3L = Phase 3 with each round range represented by its last step (rounds 2/5/10)")
    ap.add_argument("--site", default="last", choices=["last", "action"])
    ap.add_argument("--subset", default="all", help="'all', 'depth>=2' or 'depth==1' (stage-2 replication)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--t-max", type=int, default=10)
    ap.add_argument("--layers", default="", help="3X: comma-separated layers, e.g. 12,16 (all three stage schemes)")
    a = ap.parse_args()
    run, out_dir = Path(a.run), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dedupe concurrent/duplicate runs (e.g. a runner that reaches this step after it was
    # already launched by hand): wait for a live holder of the lock, then skip if done.
    tag = a.phase + (f"_L{a.layers.replace(',', '_')}" if a.phase == "3X" else "")
    lock, done = out_dir / f"probes_{tag}.lock", out_dir / f"DONE_{tag}"
    while lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except (OSError, ValueError):
            lock.unlink(missing_ok=True)
            break
        log(f"another probes --phase {a.phase} run holds {lock}; waiting")
        time.sleep(60)
    if done.exists():
        log(f"{done} exists; phase {a.phase} already complete for {out_dir}, skipping")
        return
    lock.write_text(str(os.getpid()))
    try:
        _run(a, run, out_dir)
        done.write_text(time.strftime("%F %T"))
    finally:
        lock.unlink(missing_ok=True)


def _run(a, run, out_dir):
    df = pd.read_pickle(run / "rounds.pkl")
    meta = json.loads((run / "acts" / "extraction.json").read_text())
    n_layers = meta["n_layers"]
    paths = layer_files(run, a.site, n_layers)
    keep = np.ones(len(df), bool)
    if a.subset == "depth>=2":
        keep = df["depth"].to_numpy() >= 2
    elif a.subset == "depth==1":
        keep = df["depth"].to_numpy() == 1
    elif a.subset != "all":
        raise SystemExit(f"unknown subset {a.subset}")
    rows = np.where(keep)[0]
    if not keep.all():
        # Materialise subset layer files so workers read contiguous arrays.
        sub = run / "acts" / "by_layer" / a.subset.replace(">=", "ge").replace("==", "eq")
        sub.mkdir(exist_ok=True)
        new = []
        for l, p in enumerate(paths):
            q = sub / p.name
            if not q.exists():
                np.save(q, np.load(p, mmap_mode="r")[rows])
            new.append(q)
        paths = new
    df = df.iloc[rows].reset_index(drop=True)
    folds = assign_folds(df)
    config = {"run": str(run), "site": a.site, "subset": a.subset, "n_rows": len(df),
              "n_episodes": int(df.episode_id.nunique()), "n_tasks": int(df.task_id.nunique()),
              "n_layers": n_layers, "folds": FOLDS, "seed": SEED, "phase2_t": PHASE2_T, "t_max": a.t_max,
              "labels": LABELS, "label_polarity": "1 = bad outcome", "probe_C": 1.0}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    population_table(df, a.t_max).to_csv(out_dir / "population_by_round.csv", index=False)
    log(f"{len(df)} rows, {config['n_episodes']} episodes, {config['n_tasks']} tasks, {n_layers + 1} layer slots")
    if a.phase in ("2", "all"):
        cells, _ = phase2(run, df, folds, paths, out_dir, a.jobs)
    else:
        cells = pd.read_csv(out_dir / "phase2_cells.csv")
    if a.phase in ("3", "all"):
        layers, rank = pick_layers(cells, n_layers, ruan_layer=n_layers)
        pre = out_dir / "prereg_layers.json"   # pre-registered layers are always analysed
        if pre.exists():
            layers = sorted(set(layers) | set(json.loads(pre.read_text())["layers"]))
        (out_dir / "phase3_layers.json").write_text(json.dumps(
            {"layers": layers, "rule": "top-3 mean within-task AUC over t (final, S1, logreg) + last layer (Ruan's L28 on 1.7B)",
             "cells_searched_per_label": int(len(cells) / max(1, len(LABELS))),
             "ranking": {int(k): float(v) for k, v in rank.items()}}, indent=2))
        phase3(run, df, folds, paths, layers, out_dir, a.jobs, a.t_max)
    if a.phase == "3X":
        # Explicit layers x all three stage schemes (range-averaged, milestone, last-step), both
        # populations, in ONE job set with the same folds, so layers are directly comparable.
        layers = [int(x) for x in a.layers.split(",") if x]
        schemes = ["round", "laststep"] + (["milestone"] if df["milestone"].notna().any() else [])
        jobs = [delayed(phase3_layer)(paths[l], df, folds, l, sc, pop) for l in layers for sc in schemes for pop in ("naive", "cohort")]
        log(f"phase 3X: layers {layers} x schemes {schemes} x 2 populations = {len(jobs)} jobs")
        res = Parallel(n_jobs=a.jobs)(jobs)
        pd.DataFrame([r for part in res for r in part]).to_csv(out_dir / f"phase3_transfer_L{'_'.join(map(str, layers))}.csv", index=False)
        log("phase 3X done")
    if a.phase == "3L":
        layers = json.loads((out_dir / "phase3_layers.json").read_text())["layers"]  # same layers as Phase 3
        jobs = [delayed(phase3_layer)(paths[l], df, folds, l, "laststep", pop) for l in layers for pop in ("naive", "cohort")]
        log(f"phase 3L (last step per range, rounds {LASTSTEP}): {len(jobs)} jobs on layers {layers}")
        res = Parallel(n_jobs=a.jobs)(jobs)
        pd.DataFrame([r for part in res for r in part]).to_csv(out_dir / "phase3_transfer_laststep.csv", index=False)
        log("phase 3L done")


if __name__ == "__main__":
    main()
