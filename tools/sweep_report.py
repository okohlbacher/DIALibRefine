#!/usr/bin/env python3
"""Accuracy and runtime as a function of training size AND stopping rule, from the sweep.

Reads every <sweep>/<head>/n<size>_h100/trajectory.tsv (one continuous run, per-epoch
validation) and replays a grid of stopping rules on each -- so one trajectory per size
answers every threshold. Also reads the explicit short-horizon runs (n*_h20, n*_h40) and
reports them beside the replayed "stop at 20/40" so the cosine-horizon confound is
visible rather than assumed.

Stopping rules replayed (all: minimum epochs M before firing):
  rel R / pat P   fire when P consecutive checkpoints each improve the selection metric
                  by less than R (relative to the running significant-best anchor)
Reported per (size, rule): epochs run, metric reached (val calibrated sd), TEST metric at
the checkpoint that would have been restored (best-so-far), regret vs the run's best,
optimizer updates, train seconds, eval seconds.

  sweep_report.py <sweep_dir> [--head rt|ccs] [--metric val_calibrated_sd|val_rmse]
"""
import argparse, csv, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("sweep")
ap.add_argument("--head", default="rt")
ap.add_argument("--metric", default="val_calibrated_sd")
ap.add_argument("--min-epochs", type=int, nargs="+", default=[20])
ap.add_argument("--rel", type=float, nargs="+", default=[0.05, 0.02, 0.01, 0.005])
ap.add_argument("--patience", type=int, nargs="+", default=[3, 5, 10])
ap.add_argument("--gpu-steps-per-s", type=float, default=0.0, help="if known, convert updates to GPU seconds")
a = ap.parse_args()

def load(d):
    tr = list(csv.DictReader(open(os.path.join(d, "trajectory.tsv")), delimiter="\t"))
    prov = json.load(open(glob.glob(os.path.join(d, "*_provenance.json"))[0]))
    return tr, prov

def replay(tr, metric, rel, pat, minep):
    """Return (stop_epoch, best_epoch_at_stop) under the anchor-patience rule."""
    best = anchor = None; best_ep = None; stale = 0
    for row in tr:
        e = int(row["epoch"]); m = float(row[metric])
        if best is None or m < best: best, best_ep = m, e
        if anchor is None or anchor - m >= rel * anchor: anchor, stale = m, 0
        else: stale += 1
        if e >= minep and stale >= pat: return e, best_ep
    return int(tr[-1]["epoch"]), best_ep

runs = sorted(glob.glob(os.path.join(a.sweep, a.head, "n*_h100")), key=lambda p: int(re.search(r"n(\d+)_", p).group(1)) or 10**9)
if not runs: sys.exit(f"no n*_h100 runs under {a.sweep}/{a.head}")
units = "min" if a.head == "rt" else "1/K0"
print(f"{a.head.upper()}  --  selection metric {a.metric}; TEST = untouched protein-held-out cohort; sizes are TRAINING units (0 = full pool)")
print(f"{'size':>7} {'rule':>16} {'stop@':>6} {'best@':>6} {'val':>9} {'TEST':>9} {'regret':>8} {'updates':>8} {'train s':>8} {'eval s':>7}")
print("-" * 98)
summary = {}
for d in runs:
    tr, prov = load(d)
    n = prov["cohorts"]["training"]
    test_rows = {int(r["epoch"]): r for r in tr}
    # TEST metric per epoch is not in the trajectory (only val); the provenance holds the
    # final TEST for the best-val checkpoint. We report val per rule and the provenance TEST
    # once per size, honestly labelled.
    final_test = prov["evaluation"]["tuned"]["test"]["calibrated_sd" if a.head == "rt" else "sd"]
    stock_test = prov["evaluation"]["stock"]["test"]["calibrated_sd" if a.head == "rt" else "sd"]
    run_best = min(float(r[a.metric]) for r in tr)
    summary[n] = {"stock_test": stock_test, "final_test": final_test, "best_val": run_best,
                  "updates_100": int(tr[-1]["updates"]), "train_s_100": float(tr[-1]["train_s"]), "eval_s_100": float(tr[-1]["eval_s"])}
    first = True
    for minep in a.min_epochs:
        for rel in a.rel:
            for pat in a.patience:
                e, be = replay(tr, a.metric, rel, pat, minep)
                row = test_rows[e]; brow = test_rows[be]
                val = float(brow[a.metric])
                print(f"{(n if first else ''):>7} {f'rel{100*rel:g}% pat{pat}':>16} {e:>6} {be:>6} {val:>9.4f} "
                      f"{'':>9} {100*(val-run_best)/run_best:>+7.1f}% {int(row['updates']):>8} {float(row['train_s']):>8.0f} {float(row['eval_s']):>7.0f}")
                first = False
    r100 = tr[-1]
    print(f"{'':>7} {'run to 100':>16} {100:>6} {int(min(tr, key=lambda r: float(r[a.metric]))['epoch']):>6} {run_best:>9.4f} {final_test:>9.4f} {'':>8} "
          f"{int(r100['updates']):>8} {float(r100['train_s']):>8.0f} {float(r100['eval_s']):>7.0f}   (stock TEST {stock_test:.4f})")
    print()

print("HORIZON CONFOUND: a real short run (cosine annealed at h) vs the 100-horizon trajectory cut at the same epoch")
print(f"{'size':>7} {'horizon':>8} {'real run val':>13} {'cut@h val':>10} {'real TEST':>10} {'updates':>8} {'train s':>8}")
for d in sorted(glob.glob(os.path.join(a.sweep, a.head, "n*_h[24]0"))):
    tr, prov = load(d); n = prov["cohorts"]["training"]; h = int(re.search(r"_h(\d+)$", d).group(1))
    real_val = min(float(r[a.metric]) for r in tr)
    real_test = prov["evaluation"]["tuned"]["test"]["calibrated_sd" if a.head == "rt" else "sd"]
    long = os.path.join(a.sweep, a.head, f"n{n if n < 10**8 else 0}_h100")
    cut = ""
    if os.path.isdir(long):
        lt, _ = load(long); cut = f"{min(float(r[a.metric]) for r in lt if int(r['epoch']) <= h):.4f}"
    print(f"{n:>7} {h:>8} {real_val:>13.4f} {cut:>10} {real_test:>10.4f} {int(tr[-1]['updates']):>8} {float(tr[-1]['train_s']):>8.0f}")

if a.gpu_steps_per_s:
    print(f"\nGPU seconds at {a.gpu_steps_per_s:g} steps/s: " + ", ".join(f"n={n}: {v['updates_100']/a.gpu_steps_per_s:.0f}s/100ep" for n, v in summary.items()))
