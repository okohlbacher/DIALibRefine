#!/usr/bin/env python3
"""Fine-tune AlphaPeptDeep's CCS model on one run's own measured ion mobility.

The sibling of finetune_rt.py, for the mobility head. Same loader, same
protein-level holdout, same sidecar. Two things are specific to CCS and both are
traps that would otherwise report a null result with a straight face:

1. peptdeep's ModelManager.train_ccs_model returns SILENTLY unless the frame
   carries BOTH a `mobility` and a `ccs` column (pretrained_models.py:817 --
   the two auto-conversion branches after it are unreachable). A frame with
   only mobility trains nothing, saves the stock weights, exports a stock ONNX,
   and "fine-tuning does not help CCS". So both columns are built here, and
   the saved checkpoint is asserted to differ from the stock one by hash.

2. Charge 1 is censored, not measured: S08's mobility ramp tops at 1/K0 =
   1.4013 and singly charged precursors pile against it. Training on that
   stratum teaches the model a boundary. z1 is excluded from training and from
   the headline, and reported separately.

Labels are the run's observed 1/K0 (DIA-NN report column IM). CCS is derived
from it with ODIA's own Mason-Schamp constant so that the residuals here are
in the same units as the library column they will be compared to.

  finetune_ccs.py <report.parquet> <outdir> [--epochs N] [--holdout protein|sequence]
"""
import argparse
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

# ODIA's constants (DIALibraryGenerator/include/odia/Library.h). The formula is
# 1/K0 = CCS * sqrt(mu * T) / (18509 * z); alphabase's differs by 0.0046%.
DRIFT_GAS_MASS = 28.0134
DRIFT_GAS_T = 305.0


def mobility_from_ccs(ccs, mz, z):
    m_ion = mz * z
    mu = m_ion * DRIFT_GAS_MASS / (m_ion + DRIFT_GAS_MASS)
    return ccs * np.sqrt(mu * DRIFT_GAS_T) / (18509.0 * z)


def ccs_from_mobility(k0inv, mz, z):
    m_ion = mz * z
    mu = m_ion * DRIFT_GAS_MASS / (m_ion + DRIFT_GAS_MASS)
    return k0inv * 18509.0 * z / np.sqrt(mu * DRIFT_GAS_T)


def load(path, q_max, min_charge):
    import pyarrow.parquet as pq
    index = build_unimod_index()
    per, groups, mzs = {}, {}, {}
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=200000,
                             columns=["Modified.Sequence", "Precursor.Charge", "Precursor.Mz",
                                      "IM", "Q.Value", "Protein.Group"]):
        d = b.to_pydict()
        for k in range(b.num_rows):
            if d["Q.Value"][k] > q_max or d["IM"][k] is None:
                continue
            z = int(d["Precursor.Charge"][k])
            if z < min_charge:
                continue
            key = (d["Modified.Sequence"][k], z)
            per.setdefault(key, []).append(float(d["IM"][k]))
            groups.setdefault(key, d["Protein.Group"][k] or "")
            mzs.setdefault(key, float(d["Precursor.Mz"][k]))
    rows = []
    for (seq, z), ims in per.items():
        ims.sort()
        mid = len(ims) // 2
        im = ims[mid] if len(ims) % 2 else 0.5 * (ims[mid - 1] + ims[mid])
        sequence, mods, sites = parse_modified_sequence(seq, index)
        rows.append((sequence, mods, sites, z, mzs[(seq, z)], im, groups[(seq, z)]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("identifications")
    ap.add_argument("output")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--q-value", type=float, default=0.01)
    ap.add_argument("--min-charge", type=int, default=2,
                    help="z1 is censored at the ramp top on this instrument; excluded by default")
    ap.add_argument("--holdout", choices=("sequence", "protein"), default="protein")
    ap.add_argument("--device", default="cpu", choices=("gpu", "cpu"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--curve", type=int, default=0, metavar="STEP")
    args = ap.parse_args()

    rows = load(args.identifications, args.q_value, args.min_charge)
    df = pd.DataFrame(rows, columns=["sequence", "mods", "mod_sites", "charge", "precursor_mz",
                                     "mobility", "protein_group"])
    df["nAA"] = df["sequence"].str.len()
    df["ccs"] = ccs_from_mobility(df["mobility"].values, df["precursor_mz"].values, df["charge"].values)
    print(f"{len(df):,} precursors at q <= {args.q_value}, charge >= {args.min_charge}", flush=True)

    key_col = "protein_group" if args.holdout == "protein" else "sequence"
    held = np.array([(zlib.crc32(str(s).encode()) & 0xFFFFFFFF) % 5 == 0 for s in df[key_col]])
    train_df = df[~held].reset_index(drop=True)
    print(f"held out by {args.holdout}: {int(held.sum()):,} of {len(df):,}"
          + (f" across {df.loc[held, 'protein_group'].nunique():,} protein groups" if args.holdout == "protein" else ""),
          flush=True)

    import torch
    if args.device == "cpu":
        n = args.threads or (os.cpu_count() or 1)
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 4))
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}", flush=True)

    from peptdeep.pretrained_models import ModelManager
    mgr = ModelManager(mask_modloss=False, device=args.device)
    mgr.load_installed_models()
    mgr.epoch_to_train_rt_ccs = args.epochs

    stock_pth = os.path.join(os.environ.get("ODIA_PEPTDEEP_MODELS", "/tmp/ptm/generic"), "ccs.pth")
    stock_sha = hashlib.sha256(open(stock_pth, "rb").read()).hexdigest() if os.path.exists(stock_pth) else None

    def predict_mobility(frame):
        out = mgr.ccs_model.predict(frame.copy())
        col = "ccs_pred" if "ccs_pred" in out.columns else [c for c in out.columns if c.endswith("_pred")][0]
        return mobility_from_ccs(out[col].values, frame["precursor_mz"].values, frame["charge"].values), out[col].values

    def sd_report(pred_im, label):
        r = pred_im - df["mobility"].values
        te, tr = r[held], r[~held]
        z = df["charge"].values
        by_z = {int(c): float(np.std(te[z[held] == c], ddof=1)) for c in np.unique(z) if (z[held] == c).sum() > 30}
        print(f"  {label:10} held-out: mean {te.mean():+.5f}  sd {te.std(ddof=1):.5f}   train sd {tr.std(ddof=1):.5f}   "
              f"per charge {by_z}", flush=True)
        return float(te.mean()), float(te.std(ddof=1)), by_z

    stock_im, _ = predict_mobility(df)
    s_mean, s_sd, s_z = sd_report(stock_im, "stock")

    t0 = time.time()
    if args.curve:
        done = 0
        while done < args.epochs:
            step = min(args.curve, args.epochs - done)
            mgr.epoch_to_train_rt_ccs = step
            mgr.train_ccs_model(train_df.copy())
            done += step
            pim, _ = predict_mobility(df)
            r = pim - df["mobility"].values
            print(f"  epoch {done:4d}  held-out sd {r[held].std(ddof=1):.5f}  train sd {r[~held].std(ddof=1):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    else:
        mgr.train_ccs_model(train_df.copy())
    print(f"trained in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.output, exist_ok=True)
    pth = os.path.join(args.output, "ccs.pth")
    mgr.ccs_model.save(pth)
    sha = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    if stock_sha and sha == stock_sha:
        raise SystemExit("REFUSING: the saved ccs.pth is byte-identical to the stock model. "
                         "train_ccs_model did nothing -- check that both 'mobility' and 'ccs' "
                         "columns reached it (peptdeep/pretrained_models.py:817).")

    tuned_im, _ = predict_mobility(df)
    t_mean, t_sd, t_z = sd_report(tuned_im, "tuned")

    sidecar = {
        "model": os.path.abspath(pth), "sha256": sha, "stock_sha256": stock_sha,
        "tuned_from": os.path.abspath(args.identifications),
        "precursors": len(df), "min_charge": args.min_charge, "epochs": args.epochs,
        "holdout": args.holdout, "held_out": int(held.sum()),
        "mason_schamp": {"drift_gas_mass": DRIFT_GAS_MASS, "temperature_k": DRIFT_GAS_T, "constant": 18509.0},
        "evaluation": {"stock": {"mean": s_mean, "sd": s_sd, "per_charge_sd": s_z},
                       "tuned": {"mean": t_mean, "sd": t_sd, "per_charge_sd": t_z},
                       "improvement": round(1.0 - t_sd / s_sd, 4)},
        "warning": "Specific to the run it was tuned on; the instrument's mobility calibration is baked in.",
    }
    json.dump(sidecar, open(os.path.join(args.output, "ccs_provenance.json"), "w"), indent=2)
    # per-precursor predictions, so the table can be rebuilt without re-running torch
    pd.DataFrame({"sequence": df["sequence"], "mods": df["mods"], "mod_sites": df["mod_sites"],
                  "charge": df["charge"], "held_out": held,
                  "observed_1k0": df["mobility"], "stock_1k0": stock_im, "tuned_1k0": tuned_im}
                 ).to_csv(os.path.join(args.output, "predictions.tsv"), sep="\t", index=False)
    print(f"wrote {pth} and sidecar; held-out sd {s_sd:.5f} -> {t_sd:.5f}", flush=True)


if __name__ == "__main__":
    main()
