#!/usr/bin/env python3
"""Fine-tune an AlphaPeptDeep head (RT or CCS) on one run's identifications -- the reference
implementation the C++/libtorch port must reproduce.

Supersedes finetune_rt.py / finetune_ccs.py after the 2026-09-16 review (GPT-6-Astra). Every
change below answers a finding there; the finding IDs are cited so the reasoning is traceable.

  finetune.py <report.parquet> <outdir> --head rt|ccs [options]

DESIGN, and the invariants it enforces
--------------------------------------
Cohorts are frozen on the FULL eligible data BEFORE any subsampling (C1, M2):
  test        crc32(Protein.Group) % 5 == 0          -- untouched until the end
  inner-val   crc32("val:" + Protein.Group) % 7 == 0  -- of the rest; drives stopping
  pool        everything else; a fixed seeded permutation whose PREFIXES are the nested
              training subsamples, so every --train-size shares the same test and val sets.

RT normalisation is rt / hi with lo pinned at 0 (invariant 1). hi is --rt-max-minutes, else the
maximum over the FULL eligible data -- never over a subsample (C1). Labels above hi are REJECTED and
counted, never clipped.

One continuous training call (M1). peptdeep's train_with_warmup builds ONE Adam and ONE
warmup+cosine scheduler for the whole horizon and invokes an epoch callback we install; the callback
evaluates, restores train mode (predict() switches to eval and would otherwise leave dropout off
for every later epoch), tracks the best checkpoint, and decides stopping.

Stopping (M4): patience against a SIGNIFICANT-BEST anchor -- an improvement counts only if it beats
the anchor by max(abs_tol, rel_tol * anchor); stale epochs accumulate otherwise -- with a minimum
number of epochs (never inside warmup), a hard wall-clock budget, and restoration of the best
validated state before saving. A worsened final epoch is never the model that ships.

Evaluation reports the DEPLOYED error -- hi * rt_pred against observed minutes, or 1/K0 derived
from ccs_pred with the consumer's own Mason-Schamp constants -- not only a line-fitted sd, which
cannot certify the numerical mapping (M3). The calibrated sd is reported beside it, labelled.

Charge 1 is censored at the mobility ramp top on this instrument; --min-charge 2 excludes it from
CCS training AND evaluation (invariant 4).

Data validity is a contract, not a hope (M5): finite q <= qmax (NaN q is rejected, not passed),
finite RT >= 0, finite IM > 0, finite m/z > 0, integral charge, one run.

Threads default to min(48, affinity) (M7): 128 threads on this box thrashed for 90 minutes.

The trajectory is written per epoch with enough state to replay any stopping rule offline
(stop_rule_sim.py): epoch, lr, optimizer updates, losses, every metric, train and eval seconds.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_rt import build_unimod_index, parse_modified_sequence  # noqa: E402

# ODIA's constants (DIALibraryGenerator/include/odia/Library.h:57-58,76-82).
DRIFT_GAS_MASS, DRIFT_GAS_T, MS_CONST = 28.0134, 305.0, 18509.0


def mobility_from_ccs(ccs, mz, z):
    m = mz * z
    mu = m * DRIFT_GAS_MASS / (m + DRIFT_GAS_MASS)
    return ccs * np.sqrt(mu * DRIFT_GAS_T) / (MS_CONST * z)


def ccs_from_mobility(k0inv, mz, z):
    m = mz * z
    mu = m * DRIFT_GAS_MASS / (m + DRIFT_GAS_MASS)
    return k0inv * MS_CONST * z / np.sqrt(mu * DRIFT_GAS_T)


def h5(s, salt=""):
    return (zlib.crc32((salt + str(s)).encode()) & 0xFFFFFFFF)


# --------------------------------------------------------------------------- loading
def load_report(path, head, q_max, min_charge, rt_spread_max):
    """One row per training unit, with every rejection counted.

    RT unit = modified sequence (charge states collapsed to the median RT; discordant ones rejected).
    CCS unit = (modified sequence, charge).
    """
    import pyarrow.parquet as pq
    idx = build_unimod_index()
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    need = ["Modified.Sequence", "Precursor.Charge", "Precursor.Mz", "RT", "Q.Value", "Protein.Group"]
    if head == "ccs":
        need.append("IM")
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(f"report lacks required columns {missing}")
    cols = need + (["Run"] if "Run" in have else [])

    rej = {"q_invalid": 0, "q_above": 0, "rt_invalid": 0, "im_invalid": 0, "mz_invalid": 0,
           "charge_invalid": 0, "charge_below_min": 0, "no_protein_group": 0, "decoy": 0}
    runs = set()
    obs = {}   # key -> dict(rts=[], ims=[], mz, pg, charge)
    for b in pf.iter_batches(batch_size=200000, columns=cols + (["Decoy"] if "Decoy" in have else [])):
        d = b.to_pydict()
        n = b.num_rows
        for k in range(n):
            if "Run" in d:
                runs.add(d["Run"][k])
            if "Decoy" in d and d["Decoy"][k]:
                rej["decoy"] += 1; continue
            q = d["Q.Value"][k]
            if q is None or not np.isfinite(q):
                rej["q_invalid"] += 1; continue
            if q > q_max:
                rej["q_above"] += 1; continue
            try:
                z = int(d["Precursor.Charge"][k])
            except (TypeError, ValueError):
                rej["charge_invalid"] += 1; continue
            if z < 1 or z > 8 or z != float(d["Precursor.Charge"][k]):
                rej["charge_invalid"] += 1; continue
            if z < min_charge:
                rej["charge_below_min"] += 1; continue
            mz = d["Precursor.Mz"][k]
            if mz is None or not np.isfinite(mz) or mz <= 0:
                rej["mz_invalid"] += 1; continue
            rt = d["RT"][k]
            if rt is None or not np.isfinite(rt) or rt < 0:
                rej["rt_invalid"] += 1; continue
            im = d["IM"][k] if head == "ccs" else None
            if head == "ccs" and (im is None or not np.isfinite(im) or im <= 0):
                rej["im_invalid"] += 1; continue
            pg = d["Protein.Group"][k]
            if not pg:
                rej["no_protein_group"] += 1; continue
            seq = d["Modified.Sequence"][k]
            key = seq if head == "rt" else (seq, z)
            o = obs.setdefault(key, {"rts": [], "ims": [], "mz": mz, "pg": pg, "z": z, "seq": seq})
            o["rts"].append(float(rt))
            if head == "ccs":
                o["ims"].append(float(im))
            if o["pg"] != pg:
                o["pg_conflict"] = True
    if len(runs) > 1:
        raise SystemExit(f"report contains {len(runs)} runs; fine-tuning is per run. Select one first.")

    rows, rej["rt_discordant"], rej["pg_conflict"] = [], 0, 0
    for key, o in obs.items():
        if o.get("pg_conflict"):
            rej["pg_conflict"] += 1; continue
        vals = sorted(o["rts"] if head == "rt" else o["ims"])
        if head == "rt" and len(vals) > 1 and vals[-1] - vals[0] > rt_spread_max:
            rej["rt_discordant"] += 1; continue
        mid = len(vals) // 2
        label = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
        sequence, mods, sites = parse_modified_sequence(o["seq"], idx)
        rows.append({"id": o["seq"] if head == "rt" else f"{o['seq']}/{o['z']}",
                     "sequence": sequence, "mods": mods, "mod_sites": sites, "charge": o["z"],
                     "precursor_mz": o["mz"], "protein_group": o["pg"], "label": label,
                     "n_obs": len(vals)})
    df = pd.DataFrame(rows)
    df["nAA"] = df["sequence"].str.len()
    return df, rej, (next(iter(runs)) if runs else None)


# --------------------------------------------------------------------------- metrics
def deployed_metrics(pred, truth, extra_groups=None):
    """Error of the DEPLOYED prediction against the observation, plus a labelled calibrated sd."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    if len(p) < 3:
        return {"n": int(len(p))}
    e = p - t
    a, b = np.polyfit(p, t, 1)
    cal = t - (a * p + b)
    out = {"n": int(len(p)), "mean_err": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "sd": float(e.std(ddof=1)), "p95_abs": float(np.percentile(np.abs(e), 95)),
           "calibrated_sd": float(cal.std(ddof=1)), "cal_slope": float(a), "cal_intercept": float(b)}
    if extra_groups is not None:
        g = np.asarray(extra_groups)[ok]
        out["by_group"] = {str(k): {"n": int((g == k).sum()), "sd": float(e[g == k].std(ddof=1)),
                                    "mean_err": float(e[g == k].mean())}
                           for k in np.unique(g) if (g == k).sum() >= 30}
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("identifications")
    ap.add_argument("output")
    ap.add_argument("--head", choices=("rt", "ccs"), required=True)
    ap.add_argument("--q-value", type=float, default=0.01)
    ap.add_argument("--min-charge", type=int, default=2, help="CCS only. 1 is censored here; refused unless --allow-z1")
    ap.add_argument("--allow-z1", action="store_true")
    ap.add_argument("--rt-spread-max", type=float, default=0.2, help="RT: reject sequences whose charge states disagree by more (minutes)")
    ap.add_argument("--rt-max-minutes", type=float, default=0.0, help="RT scale. 0 = max over the FULL eligible data")
    ap.add_argument("--train-size", type=int, default=0, help="training units from the pool (0 = all); nested prefixes")
    ap.add_argument("--train-frac", type=float, default=0.0, help="alternative to --train-size")
    ap.add_argument("--no-inner-val", action="store_true", help="drive stopping on the test fifth (then it is validation, not test)")
    ap.add_argument("--epochs", type=int, default=100, help="horizon of the ONE schedule; stopping may end earlier")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=10, help="stale epochs before stopping")
    ap.add_argument("--rel-tol", type=float, default=0.005)
    ap.add_argument("--abs-tol", type=float, default=0.0, help="in the selection metric's units")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="hard wall budget for training (0 = none)")
    ap.add_argument("--select", choices=("rmse", "calibrated_sd"), default="calibrated_sd",
                    help="checkpoint selection metric. calibrated_sd is what a library consumer sees -- both DIA-NN "
                         "and ODIA refit RT per run -- so it is the default; rmse is the deployed mapping error and "
                         "is always recorded beside it")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260803)
    a = ap.parse_args()
    if a.head == "ccs" and a.min_charge < 2 and not a.allow_z1:
        raise SystemExit("--min-charge 1 for CCS: z1 is censored at the ramp top on this instrument; pass --allow-z1 to override")
    if a.epochs <= 0 or a.eval_every <= 0 or a.patience <= 0 or a.rel_tol < 0 or a.abs_tol < 0:
        raise SystemExit("epochs, eval-every, patience must be > 0; tolerances >= 0")

    t0 = time.time()
    df, rej, run = load_report(a.identifications, a.head, a.q_value,
                               a.min_charge if a.head == "ccs" else 1, a.rt_spread_max)
    if len(df) < 200:
        raise SystemExit(f"only {len(df)} usable units after validation; rejections: {rej}")
    print(f"{len(df):,} eligible {a.head} units from run {run!r}; rejected {rej}", flush=True)

    # ---- scale (RT) or derived target (CCS), on the FULL eligible data
    if a.head == "rt":
        hi = a.rt_max_minutes if a.rt_max_minutes > 0 else float(df["label"].max())
        over = int((df["label"] > hi).sum())
        if over:
            raise SystemExit(f"{over} units have RT above the declared scale {hi} min; refusing to clip. "
                             f"Set --rt-max-minutes to the run's gradient length.")
        df["rt"] = df["label"]
        df["rt_norm"] = df["label"] / hi
        scale = {"rt_norm_min_minutes": 0.0, "rt_norm_max_minutes": hi}
    else:
        hi = None
        df["mobility"] = df["label"]
        df["ccs"] = ccs_from_mobility(df["label"].values, df["precursor_mz"].values, df["charge"].values)
        scale = {"mason_schamp": {"drift_gas_mass": DRIFT_GAS_MASS, "temperature_k": DRIFT_GAS_T, "constant": MS_CONST}}

    # ---- cohorts: frozen on the full data, before any subsampling
    test = np.array([h5(g) % 5 == 0 for g in df["protein_group"]])
    val = np.array([(not t) and h5(g, "val:") % 7 == 0 for t, g in zip(test, df["protein_group"])]) \
        if not a.no_inner_val else test.copy()
    pool = ~(test | val) if not a.no_inner_val else ~test
    rng = np.random.default_rng(a.seed)
    order = np.flatnonzero(pool)[rng.permutation(int(pool.sum()))]
    n_train = int(pool.sum())
    if a.train_frac > 0:
        n_train = max(1, int(round(a.train_frac * n_train)))
    if a.train_size > 0:
        n_train = min(n_train, a.train_size)
    train_idx = order[:n_train]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df, test_df = df[val].reset_index(drop=True), df[test].reset_index(drop=True)
    print(f"cohorts: pool {int(pool.sum()):,} -> training {n_train:,} | inner-val {len(val_df):,} "
          f"({val_df['protein_group'].nunique():,} groups) | test {len(test_df):,} "
          f"({test_df['protein_group'].nunique():,} groups){'  [val == test]' if a.no_inner_val else ''}",
          flush=True)
    if len(val_df) < 100 or len(test_df) < 100:
        raise SystemExit("validation or test cohort too small")

    # ---- torch, seeds, threads
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    use_cuda = a.device == "gpu" and torch.cuda.is_available()
    if a.device == "gpu" and not use_cuda:
        print("WARNING: --device gpu requested but CUDA is unavailable; running on CPU", file=sys.stderr)
    # peptdeep/model/building_block.py calls torch.set_num_threads() at IMPORT time and
    # silently drops the count (to 2 here), so the thread count must be set AFTER the
    # import, not before. Measured: 24 requested, 2 in effect, until this ordering.
    from peptdeep.pretrained_models import ModelManager
    from peptdeep.model.model_interface import CallbackHandler
    if not use_cuda:
        n_thr = a.threads or min(48, len(os.sched_getaffinity(0)))
        torch.set_num_threads(n_thr)
    mgr = ModelManager(mask_modloss=False, device="gpu" if use_cuda else "cpu")
    mgr.load_installed_models()
    model = mgr.rt_model if a.head == "rt" else mgr.ccs_model
    dev = next(model.model.parameters()).device
    print(f"torch {torch.__version__} device {dev} threads {torch.get_num_threads()}", flush=True)

    def predict(frame):
        out = model.predict(frame.copy())
        if a.head == "rt":
            return hi * out["rt_pred"].values
        return mobility_from_ccs(out["ccs_pred"].values, frame["precursor_mz"].values, frame["charge"].values)

    def evaluate(frame):
        return deployed_metrics(predict(frame), frame["label"].values,
                                frame["charge"].values if a.head == "ccs" else None)

    # ---- stock baseline, and the tensors we will prove changed
    stock_state = copy.deepcopy(model.model.state_dict())
    stock_val, stock_test = evaluate(val_df), evaluate(test_df)
    print(f"stock   val {a.select} {stock_val[a.select]:.5f}   test {a.select} {stock_test[a.select]:.5f}", flush=True)

    # ---- the callback: evaluate, restore train mode, track best, decide stopping
    traj_path = os.path.join(a.output, "trajectory.tsv")
    os.makedirs(a.output, exist_ok=True)
    traj = open(traj_path, "w")
    traj.write("epoch\tlr\tupdates\ttrain_loss\tval_rmse\tval_sd\tval_mean_err\tval_p95\tval_calibrated_sd\t"
               "train_s\teval_s\tbest\tstop\n")

    class Stopper(CallbackHandler):
        def __init__(s):
            s.updates = 0; s.best = None; s.best_epoch = -1; s.best_state = None
            s.anchor = None; s.stale = 0; s.t_start = time.time(); s.t_eval = 0.0; s.reason = "horizon"

        def batch_callback(s, batch, batch_loss):
            s.updates += 1

        def epoch_callback(s, epoch, epoch_loss):
            e = epoch + 1
            if e % a.eval_every and e != a.epochs:
                return True
            te = time.time()
            m = evaluate(val_df)
            model.model.train()          # predict() left it in eval mode
            s.t_eval += time.time() - te
            metric = m[a.select]
            improved = s.best is None or metric < s.best
            if improved:
                s.best, s.best_epoch = metric, e
                s.best_state = copy.deepcopy(model.model.state_dict())
            eps = max(a.abs_tol, a.rel_tol * (s.anchor if s.anchor is not None else metric))
            if s.anchor is None or s.anchor - metric >= eps:
                s.anchor, s.stale = metric, 0
            else:
                s.stale += a.eval_every
            lr = model.optimizer.param_groups[0]["lr"]
            train_s = time.time() - s.t_start - s.t_eval
            stop = ""
            if not np.isfinite(metric):
                stop, s.reason = "nonfinite", "nonfinite metric"
            elif a.max_seconds and train_s > a.max_seconds:
                stop, s.reason = "budget", f"wall budget {a.max_seconds}s"
            elif e >= max(a.min_epochs, a.warmup) and s.stale >= a.patience:
                stop, s.reason = "patience", f"no significant improvement for {s.stale} epochs"
            traj.write(f"{e}\t{lr:.3e}\t{s.updates}\t{epoch_loss:.6f}\t{m['rmse']:.6f}\t{m['sd']:.6f}\t"
                       f"{m['mean_err']:+.6f}\t{m['p95_abs']:.6f}\t{m['calibrated_sd']:.6f}\t"
                       f"{train_s:.1f}\t{s.t_eval:.1f}\t{'*' if improved else ''}\t{stop}\n"); traj.flush()
            print(f"  epoch {e:4d}  lr {lr:.2e}  loss {epoch_loss:.5f}  val {a.select} {metric:.5f}"
                  f"{' *' if improved else ''}  stale {s.stale}  {train_s:.0f}s{('  STOP: ' + s.reason) if stop else ''}",
                  flush=True)
            return not stop

    cb = Stopper()
    model.set_callback_handler(cb)

    # ---- ONE training call; peptdeep builds one optimizer and one warmup+cosine schedule for it
    mgr.epoch_to_train_rt_ccs = a.epochs
    mgr.warmup_epoch_to_train_rt_ccs = a.warmup
    mgr.lr_to_train_rt_ccs = a.lr
    mgr.batch_size_to_train_rt_ccs = a.batch_size
    t_train = time.time()
    if a.head == "rt":
        mgr.train_rt_model(train_df.copy())
    else:
        mgr.train_ccs_model(train_df.copy())
    train_wall = time.time() - t_train
    traj.close()

    # ---- guards: it actually trained, and the weights actually moved (M6)
    if cb.updates == 0:
        raise SystemExit("REFUSING: zero optimizer updates -- peptdeep trained nothing (missing target columns?)")
    if cb.best_state is None:
        raise SystemExit("REFUSING: no evaluated checkpoint")
    model.model.load_state_dict(cb.best_state)      # the model that ships is the best VALIDATED one
    moved = sum(float(((cb.best_state[k].float() - stock_state[k].float()) ** 2).sum()) for k in stock_state) ** 0.5
    if moved == 0.0:
        raise SystemExit("REFUSING: parameters are byte-identical to stock after training")
    if not all(torch.isfinite(v).all() for v in cb.best_state.values()):
        raise SystemExit("REFUSING: non-finite parameters")

    # ---- final evaluation on val (drove stopping) and TEST (untouched)
    tuned_val, tuned_test = evaluate(val_df), evaluate(test_df)
    pth = os.path.join(a.output, f"{a.head}.pth")
    model.save(pth)
    digest = hashlib.sha256(open(pth, "rb").read()).hexdigest()

    pred_test = predict(test_df)
    pd.DataFrame({"id": test_df["id"], "protein_group": test_df["protein_group"], "charge": test_df["charge"],
                  "observed": test_df["label"], "stock": predict_stock(model, stock_state, test_df, predict),
                  "tuned": pred_test}).to_csv(os.path.join(a.output, "test_predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(a.output, "training_ids.txt"), "w") as fh:
        fh.write("\n".join(train_df["id"]) + "\n")

    side = {
        "head": a.head, "model": os.path.abspath(pth), "sha256": digest, "run": run,
        "tuned_from": os.path.abspath(a.identifications), "q_value": a.q_value,
        "rejections": rej, "eligible_units": int(len(df)),
        "cohorts": {"training": int(n_train), "pool": int(pool.sum()), "inner_val": int(len(val_df)),
                    "test": int(len(test_df)), "val_is_test": bool(a.no_inner_val),
                    "test_rule": "crc32(Protein.Group)%5==0", "val_rule": "crc32('val:'+Protein.Group)%7==0 of the rest",
                    "subsample": "fixed seeded permutation of the pool, nested prefixes", "seed": a.seed},
        "recipe": {"epochs_horizon": a.epochs, "warmup_epochs": a.warmup, "lr": a.lr, "batch_size": a.batch_size,
                   "loss": "L1 (peptdeep default)", "optimizer": "Adam, one instance", "schedule": "linear warmup then cosine, one instance",
                   "eval_every": a.eval_every, "min_epochs": a.min_epochs, "patience": a.patience,
                   "rel_tol": a.rel_tol, "abs_tol": a.abs_tol, "max_seconds": a.max_seconds, "select": a.select,
                   "min_charge": a.min_charge if a.head == "ccs" else None, "rt_spread_max": a.rt_spread_max},
        **scale,
        "training": {"epochs_completed": int(cb.best_epoch if cb.reason != "horizon" else a.epochs),
                     "best_epoch": cb.best_epoch, "stop_reason": cb.reason, "optimizer_updates": cb.updates,
                     "parameter_change_l2": moved, "train_seconds": round(train_wall - cb.t_eval, 1),
                     "eval_seconds": round(cb.t_eval, 1), "device": str(dev), "threads": torch.get_num_threads(),
                     "torch": torch.__version__},
        "evaluation": {"stock": {"val": stock_val, "test": stock_test}, "tuned": {"val": tuned_val, "test": tuned_test},
                       "units": "minutes (deployed hi*rt_pred)" if a.head == "rt" else "1/K0 (Vs/cm2) from ccs_pred",
                       "improvement_test_rmse": round(1 - tuned_test["rmse"] / stock_test["rmse"], 4)},
        "warning": "Per-run model: the run's RT scale / mobility calibration is baked in.",
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(side, open(os.path.join(a.output, f"{a.head}_provenance.json"), "w"), indent=2)
    print(f"tuned   val {a.select} {tuned_val[a.select]:.5f}   TEST rmse {stock_test['rmse']:.5f} -> {tuned_test['rmse']:.5f}"
          f"  (calibrated sd {stock_test['calibrated_sd']:.5f} -> {tuned_test['calibrated_sd']:.5f})"
          f"  best epoch {cb.best_epoch}, {cb.updates} updates, {train_wall:.0f}s; stop: {cb.reason}", flush=True)


def predict_stock(model, stock_state, frame, predict):
    """Score with the stock weights, then put the tuned weights back."""
    tuned = copy.deepcopy(model.model.state_dict())
    model.model.load_state_dict(stock_state)
    out = predict(frame)
    model.model.load_state_dict(tuned)
    return out


if __name__ == "__main__":
    main()
