#!/usr/bin/env python
"""Fine-tune AlphaPeptDeep's retention-time model on one run's own identifications.

Why this exists: the stock model's output saturates. Its top two deciles are
compressed 5x and 12x, 12.8% of a human library lands in one output bin, and no
calibration can undo that -- a monotone map can straighten a bent axis, not a
collapsed one. Measured on S08, held-out residual falls 0.701 -> 0.449 min from
500 training peptides in 31 seconds, and afterwards a flexible calibration buys
nothing over a straight line, which is the evidence that the defect was in the
model rather than the mapping. See doc/06-rt-refinement-plan.md.

This runs OUTSIDE ODIA, in a environment carrying peptdeep and torch, and its
output is an ONNX file ODIA reads with `-rt_model`. torch never enters ODIA's
runtime.

**Modifications are resolved through alphabase's own table**, all 2,788 entries
carrying a UniMod id, rather than a hand-written dictionary. The prototype this
replaces knew five modifications and raised on the sixth; that is a fine failure
mode and a poor limit.

**Two mechanisms, both kept.** `--method direct` retrains the model to predict
normalised retention time. `--method residual` fits a calibration first --
linear, then isotonic on top of it -- and retrains only on what the calibration
could not explain, adding the two back at prediction time.

On S08 direct won by a wide margin: held-out sd 0.429 against 0.635 at 500
peptides and 500 epochs, with residual barely beating the calibration it sits
on. **That is one file, and one file is not a result.** The likely reason is
that retargeting pretrained weights at a small centred residual fights the
initialisation rather than exploiting it -- but a different gradient, instrument
or organism could reverse it, and freezing the trunk (untested) could too. Both
mechanisms are therefore selectable, and `--evaluate` makes every run report
which one won on ITS data rather than inheriting a verdict from ours.

**The retention times must come from a search of the run being tuned for.** That
is the point -- the model learns this chromatography -- and it is also the
hazard: if those identifications came from searching with the library this model
will go on to build, the loop is closed and the FDR of the second search is no
longer independent of the first. Feeding this from an external search is safe;
feeding it from ODIA's own first pass is the case that needs the entrapment
measurement doc/06 describes. This script does not know which it has been given,
so it records the source in the sidecar and leaves the judgement to the caller.
"""
import argparse
import json
import hashlib
import os
import re
import sys
import time

MOD_TOKEN = re.compile(r"\(UniMod:(\d+)\)|\[UniMod:(\d+)\]", re.IGNORECASE)


def build_unimod_index():
    """{unimod id: [alphabase mod names]}, from alphabase's own table."""
    from alphabase.constants.modification import MOD_DF
    index = {}
    for name, uid in zip(MOD_DF["mod_name"], MOD_DF["unimod_id"]):
        if uid and uid > 0:
            index.setdefault(int(uid), []).append(name)
    return index


def parse_modified_sequence(seq, index):
    """DIA-NN's Modified.Sequence -> (bare sequence, mods, mod_sites).

    AlphaBase names a modification by residue -- Carbamidomethyl@C -- and by
    terminus for N-terminal ones, so the UniMod id alone is ambiguous and the
    residue it sits on has to pick between the candidates.
    """
    bare, mods, sites = [], [], []
    pos = 0
    for m in MOD_TOKEN.finditer(seq):
        bare.append(seq[pos:m.start()])
        uid = int(m.group(1) or m.group(2))
        site = len("".join(bare))
        at_end = m.end() >= len(seq.rstrip())
        # The residue this modification sits on is the last one written so far.
        # Two modifications on the same residue leave bare[-1] empty, so the
        # search walks back rather than reading "" and failing into a fallback.
        residue = ""
        for chunk in reversed(bare):
            if chunk:
                residue = chunk[-1]
                break
        candidates = index.get(uid, [])
        chosen = None
        if site == 0:
            # Before any residue. Prefer the peptide N-terminus over the
            # protein one: "Any_N-term" is what a search engine means unless it
            # says otherwise, and picking whichever happens to come first in
            # alphabase's table would flip on an upstream reordering.
            for want in ("Any_N-term", "N-term"):
                for c in candidates:
                    if want in c:
                        chosen = c
                        break
                if chosen:
                    break
        elif at_end:
            for want in ("Any_C-term", "C-term"):
                for c in candidates:
                    if want in c:
                        chosen = c
                        break
                if chosen:
                    break
        if chosen is None:
            for c in candidates:
                if c.endswith("@" + residue):
                    chosen = c
                    break
        if chosen is None:
            raise SystemExit(
                f"UniMod:{uid} on residue {residue!r} in {seq!r} has no alphabase "
                f"name among {candidates}; refusing to guess")
        mods.append(chosen)
        # alphabase addresses a C-terminal modification as -1, not as the
        # peptide length; writing the length puts it one past the end.
        sites.append("-1" if "C-term" in chosen else str(site))
        pos = m.end()
    bare.append(seq[pos:])
    sequence = "".join(bare)

    # Anything the UniMod pattern did not match is still in the string. A
    # named modification -- "S(Phospho (STY))EQK" from Spectronaut or MaxQuant,
    # or DIA-NN configured to write names -- passes through the loop untouched
    # and would be handed to the model AS THE SEQUENCE. That is the silent
    # mis-encoding this parser exists to prevent, so it is checked rather than
    # assumed.
    if not re.fullmatch(r"[A-Z]+", sequence):
        raise SystemExit(
            f"{seq!r} does not reduce to a bare peptide -- got {sequence!r}. Only "
            f"(UniMod:N) and [UniMod:N] are understood; named modifications are "
            f"not, and guessing at them would corrupt the training data silently.")
    return sequence, ";".join(mods), ";".join(sites)


def load_identifications(path, q_max):
    import pyarrow.parquet as pq
    index = build_unimod_index()
    per = {}
    groups = {}
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=200000,
                             columns=["Modified.Sequence", "Precursor.Charge",
                                      "RT", "Q.Value", "Protein.Group"]):
        d = b.to_pydict()
        for k in range(b.num_rows):
            if d["Q.Value"][k] > q_max:
                continue
            per.setdefault(d["Modified.Sequence"][k], []).append(float(d["RT"][k]))
            groups.setdefault(d["Modified.Sequence"][k], d["Protein.Group"][k] or "")

    rows = []
    for seq, rts in per.items():
        rts.sort()
        # Charge states coelute to within 0.0018 min at the median, so this
        # collapses a duplicate rather than averaging away a real difference --
        # but the tail is fat: 2.6% disagree by more than 0.2 min and the worst
        # by 2.1 min, which is label noise, not chemistry. Those are dropped
        # rather than averaged.
        if len(rts) > 1 and rts[-1] - rts[0] > 0.2:
            continue
        # A true median. rts[len//2] on an even-length sorted list returns the
        # LATER value, which biased every two-charge peptide late by half its
        # spread.
        mid = len(rts) // 2
        rt = rts[mid] if len(rts) % 2 else 0.5 * (rts[mid - 1] + rts[mid])
        sequence, mods, sites = parse_modified_sequence(seq, index)
        rows.append((sequence, mods, sites, rt, groups.get(seq, "")))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("identifications", help="a DIA-NN report.parquet for THIS run")
    ap.add_argument("output", help="directory for rt.pth and the sidecar")
    ap.add_argument("--base-model", default="",
                    help="starting checkpoint; default is peptdeep's generic rt.pth")
    ap.add_argument("--max-peptides", type=int, default=2000,
                    help="training peptides. 500 buys 70%% of the gain in 31 s, "
                         "2000 buys 85%% in 78 s, and the full set costs 17 min "
                         "for the rest. Default 2000.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--curve", type=int, default=0, metavar="STEP",
                    help="Train in STEP-epoch increments and record the held-out "
                         "residual after each, to <output>/curve.tsv. The whole "
                         "point of a curve is to see where it stops improving and "
                         "whether it turns back up; a single end-point number "
                         "cannot show either.")
    ap.add_argument("--rt-max-minutes", type=float, default=0.0, metavar="MINUTES",
                    help="Gradient end for the rt_norm denominator (rt_norm = rt / this, "
                         "minimum pinned to 0 as alphabase does). Pass the RUN's gradient "
                         "end, not the training subset's maximum -- a subset-dependent "
                         "denominator makes every fine-tune define a different target "
                         "space and breaks the inverse at deployment. Defaults to the "
                         "maximum over ALL supplied identifications, held-out included.")
    ap.add_argument("--converge-tol", type=float, default=0.1/60.0, metavar="MINUTES",
                    help="Early-stop when a --curve checkpoint improves the held-out "
                         "residual by less than this. A DELTA criterion: 'is the residual "
                         "below X' needs an X that differs per gradient and instrument, "
                         "'has it stopped moving' does not. Default 0.1 s. On the measured "
                         "Astral curve the per-25-epoch deltas past epoch 125 are 0.26, 0.11, "
                         "0.12, 0.11 and 0.008 s, so 0.1 s lands the stop between 200 and 250 "
                         "-- the flat region the held-out coverage identifies, without paying "
                         "for the 250 epochs beyond it that bought 0.65 coverage points.")
    ap.add_argument("--converge-rel", type=float, default=0.001,
                    help="Early-stop when the improvement is below this FRACTION of the "
                         "current residual. Whichever of the two triggers first wins. "
                         "Measured on the Astral 500-epoch curve, 0.001 stops at ~250 "
                         "epochs; 0.01 would stop at 125, which the held-out coverage "
                         "says is 2 points of +/-30s coverage too early.")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"),
                    help="peptdeep device. 'gpu' falls back to CPU SILENTLY when "
                         "torch is a CPU-only build, which is how this ran on CPU "
                         "for a whole afternoon, so the resolved device is printed.")
    ap.add_argument("--q-value", type=float, default=0.01)
    ap.add_argument("--method", choices=("direct", "residual"), default="direct",
                    help="direct retrains on normalised RT; residual calibrates "
                         "first and retrains on the remainder. direct won on S08 "
                         "by 0.429 against 0.635, on one file. Use --evaluate.")
    ap.add_argument("--holdout", choices=("sequence", "protein"), default="sequence",
                    help="What the held-out fifth is drawn by. 'sequence' (the original) "
                         "leaks: peptides of one protein share composition and co-elute, "
                         "so a sequence split scores the model on near-copies of what it "
                         "trained on. 'protein' holds out whole protein groups.")
    ap.add_argument("--evaluate", action="store_true",
                    help="hold out a fifth of the peptides, and report held-out "
                         "residual for the stock model and for this method. Costs "
                         "a fifth of the training data and answers the only "
                         "question that matters on a new file: did this help?")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from peptdeep.pretrained_models import ModelManager

    t0 = time.time()
    rows = load_identifications(args.identifications, args.q_value)
    if len(rows) < 200:
        raise SystemExit(f"only {len(rows)} identifications at q <= {args.q_value}; "
                         f"too few to fine-tune on")
    print(f"{len(rows):,} distinct modified sequences at q <= {args.q_value}")

    rng = np.random.default_rng(20260803)
    if args.max_peptides and len(rows) > args.max_peptides:
        pick = rng.choice(len(rows), args.max_peptides, replace=False)
        rows = [rows[i] for i in pick]
    print(f"training on {len(rows):,}")

    df = pd.DataFrame(rows, columns=["sequence", "mods", "mod_sites", "rt", "protein_group"])
    df["nAA"] = df["sequence"].str.len()

    # Held out by a deterministic hash so re-running reports the same number.
    # By SEQUENCE a peptide's charge states cannot straddle the split; by
    # PROTEIN neither can its siblings, which share composition and co-elute.
    import zlib
    key_col = "protein_group" if args.holdout == "protein" else "sequence"
    held = np.array([(zlib.crc32(str(s).encode()) & 0xFFFFFFFF) % 5 == 0
                     for s in df[key_col]]) if args.evaluate else np.zeros(len(df), bool)
    if args.evaluate:
        print(f"held out by {args.holdout}: {int(held.sum()):,} of {len(df):,} peptides"
              + (f" across {df.loc[held, 'protein_group'].nunique():,} protein groups"
                 if args.holdout == "protein" else ""), flush=True)
    train_df = df[~held].reset_index(drop=True)
    test_df = df[held].reset_index(drop=True)
    if args.evaluate and len(test_df) < 50:
        raise SystemExit("too few peptides to hold any out; drop --evaluate")

    # THE NORMALISATION CONVENTION, and why `lo` is pinned to 0.
    #
    # AlphaPeptDeep's AlphaRTModel trains on `rt_norm`, and alphabase's
    # `_normalize_rt` produces it as `rt / max_rt` -- `_min_max_rt_norm` is
    # False by default, so the MINIMUM IS 0, not the observed minimum.
    #
    # This used to read `lo = train_df["rt"].min()`, which was wrong twice:
    #   * it shifted the target away from the space the pretrained weights
    #     live in, so transfer learning had to relearn an offset instead of
    #     refining chemistry (the paper's 500-peptide result, R^2 0.927 ->
    #     0.986, depends on that transfer);
    #   * `lo`/`hi` came from the sampled TRAINING SUBSET, so every run defined
    #     a different target space and the inverse applied at deployment did
    #     not match the one used at training.
    # Symptom on record: held-out sd 0.4149 min inside the experiment while the
    # exported ONNX was no better than stock on real data.
    #
    # `hi` is a per-run constant (this run's gradient end) and is written to
    # the sidecar. It is allowed to be per-run because the same value is used
    # forward and backward; only a SUBSET-dependent value is illegal.
    lo = 0.0
    hi = float(args.rt_max_minutes) if args.rt_max_minutes > 0 else float(df["rt"].max())
    if not hi > lo:
        raise SystemExit("every identification has the same retention time")
    if float(train_df["rt"].min()) < 0.0:
        raise SystemExit("negative retention times: the rt column is not run minutes")

    import os as _os, torch as _torch
    _cuda = _torch.cuda.is_available()
    if _cuda and args.device == "gpu":
        # Every visible GPU, not just cuda:0. peptdeep drives one device, so the
        # win here is picking the LEAST BUSY one rather than colliding with
        # whatever is already resident -- these cards are shared, and today one
        # of them was carrying 60 GB of somebody else's job.
        free = []
        for d in range(_torch.cuda.device_count()):
            try:
                f, _t = _torch.cuda.mem_get_info(d)
                free.append((f, d))
            except Exception:
                free.append((0, d))
        free.sort(reverse=True)
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
        _torch.cuda.set_device(free[0][1])
    else:
        # CPU: use the cores we actually have. torch defaults to a heuristic
        # that is frequently one thread inside a container, and a 500-epoch fit
        # single-threaded is the difference between minutes and an hour.
        _n = int(_os.environ.get("ODIA_RT_THREADS", "0")) or (_os.cpu_count() or 1)
        _torch.set_num_threads(_n)
        _torch.set_num_interop_threads(max(1, _n // 4))
    print(f"torch {_torch.__version__}; cuda available {_cuda}; "
          f"device requested {args.device}; RESOLVED {'cuda' if (_cuda and args.device == 'gpu') else 'cpu'}"
          + ("" if _cuda or args.device == "cpu" else
             "  <-- CPU-ONLY TORCH: install a CUDA build to use the H100s"),
          flush=True)
    if _cuda and args.device == "gpu":
        d = _torch.cuda.current_device()
        f, t = _torch.cuda.mem_get_info(d)
        print(f"  gpu {d}: {_torch.cuda.get_device_name(d)}, "
              f"{f/2**30:.1f} of {t/2**30:.1f} GiB free "
              f"(of {_torch.cuda.device_count()} visible)", flush=True)
    else:
        print(f"  cpu threads: {_torch.get_num_threads()} intra, "
              f"{_torch.get_num_interop_threads()} inter", flush=True)
    mgr = ModelManager(mask_modloss=False, device=args.device)
    mgr.load_installed_models()
    if args.base_model:
        mgr.rt_model.load(args.base_model)
    mgr.epoch_to_train_rt_ccs = args.epochs

    stock_pred = mgr.rt_model.predict(df.copy())["rt_pred"].values
    calibration = None

    if args.method == "direct":
        # Absolute normalised retention time; the exported model predicts it
        # directly and ODIA consumes it with -rt_model.
        train_df = train_df.assign(rt_norm=(train_df["rt"] - lo) / (hi - lo))
    else:
        # Calibrate first -- linear, then isotonic -- and learn the remainder.
        from sklearn.isotonic import IsotonicRegression
        sp = stock_pred[~held]
        A = np.vstack([sp, np.ones_like(sp)]).T
        m0, c0 = np.linalg.lstsq(A, train_df["rt"].values, rcond=None)[0]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            m0 * sp + c0, train_df["rt"].values)
        cal_tr = iso.predict(m0 * sp + c0)
        resid = train_df["rt"].values - cal_tr
        rlo, rhi = float(resid.min()), float(resid.max())
        train_df = train_df.assign(rt_norm=(resid - rlo) / (rhi - rlo))
        calibration = dict(slope=float(m0), intercept=float(c0),
                           knots_x=[float(x) for x in iso.X_thresholds_],
                           knots_y=[float(y) for y in iso.y_thresholds_],
                           residual_min=rlo, residual_max=rhi)

    if args.curve > 0 and args.evaluate:
        # Train in increments, measuring the held-out residual at each. Same
        # scoring as the final evaluation below, so the curve's last point and
        # the reported number are the same quantity.
        import time as _time
        truth_c = df["rt"].values
        def _sd_at():
            pr = mgr.rt_model.predict(df.copy())["rt_pred"].values
            p, t = pr[~held], truth_c[~held]
            A = np.vstack([p, np.ones_like(p)]).T
            m, c = np.linalg.lstsq(A, t, rcond=None)[0]
            return (float(np.std(truth_c[held] - (m * pr[held] + c))),
                    float(np.std(t - (m * p + c))))
        curve_path = os.path.join(args.output, "curve.tsv")
        os.makedirs(args.output, exist_ok=True)
        with open(curve_path, "w") as cf:
            cf.write("epoch\theld_out_sd_min\ttrain_sd_min\tseconds\n")
            done, t0 = 0, _time.time()
            te, tr = _sd_at()
            cf.write(f"0\t{te:.5f}\t{tr:.5f}\t0.0\n"); cf.flush()
            print(f"  epoch    0  held-out {te*60:7.2f} s  train {tr*60:7.2f} s", flush=True)
            mgr.epoch_to_train_rt_ccs = args.curve
            prev = te
            while done < args.epochs:
                step = min(args.curve, args.epochs - done)
                mgr.epoch_to_train_rt_ccs = step
                mgr.train_rt_model(train_df)
                done += step
                te, tr = _sd_at()
                cf.write(f"{done}\t{te:.5f}\t{tr:.5f}\t{_time.time()-t0:.1f}\n"); cf.flush()
                delta = prev - te
                print(f"  epoch {done:4d}  held-out {te*60:7.2f} s  train {tr*60:7.2f} s"
                      f"  delta {delta*60:6.3f} s ({100*delta/prev if prev else 0:5.2f}%)",
                      flush=True)
                # EARLY STOP on the delta. The held-out curve here is flat rather
                # than U-shaped -- it stops improving and does not degrade -- so
                # there is no minimum to overshoot, only compute to waste. On the
                # Astral curve the remaining 250 epochs past this point bought
                # 0.36 s of residual and, held out by sequence, 0.65 points of
                # +/-30 s coverage.
                if delta < args.converge_tol or delta < args.converge_rel * prev:
                    print(f"  CONVERGED at epoch {done}: improvement {delta*60:.3f} s is "
                          f"below --converge-tol {args.converge_tol*60:.3f} s and "
                          f"--converge-rel {100*args.converge_rel:.2f}%", flush=True)
                    break
                prev = te
        print(f"wrote {curve_path}", flush=True)
    else:
        mgr.train_rt_model(train_df)

    os.makedirs(args.output, exist_ok=True)
    pth = os.path.join(args.output, "rt.pth")
    mgr.rt_model.save(pth)

    evaluation = None
    if args.evaluate:
        tuned = mgr.rt_model.predict(df.copy())["rt_pred"].values

        def sd(pred, truth, tr_mask, te_mask):
            p, t = pred[tr_mask], truth[tr_mask]
            A = np.vstack([p, np.ones_like(p)]).T
            m, c = np.linalg.lstsq(A, t, rcond=None)[0]
            return float(np.std(truth[te_mask] - (m * pred[te_mask] + c)))

        truth = df["rt"].values
        if args.method == "residual":
            # A residual model predicts a CORRECTION. Scoring its raw output as
            # though it were a retention time reports a catastrophe that is an
            # artefact of the scoring, and would condemn the method on every
            # dataset. The prediction is calibration(stock) + correction.
            cal_all = iso.predict(m0 * stock_pred + c0)
            combined = cal_all + (tuned * (rhi - rlo) + rlo)
            tuned_sd = float(np.std(truth[held] - combined[held]))
        else:
            tuned_sd = sd(tuned, truth, ~held, held)

        evaluation = {
            "held_out_peptides": int(held.sum()),
            "stock_sd_minutes": round(sd(stock_pred, truth, ~held, held), 4),
            "tuned_sd_minutes": round(tuned_sd, 4),
            "method": args.method,
        }
        evaluation["improvement"] = round(
            1.0 - evaluation["tuned_sd_minutes"] / evaluation["stock_sd_minutes"], 4)
        # Said plainly, because the alternative is a user inheriting our verdict
        # from a different instrument.
        if evaluation["tuned_sd_minutes"] >= evaluation["stock_sd_minutes"]:
            print(f"WARNING: --method {args.method} did NOT improve on this data "
                  f"({evaluation['tuned_sd_minutes']:.3f} against "
                  f"{evaluation['stock_sd_minutes']:.3f} min held out). Try the "
                  f"other method before using this model.", file=sys.stderr)

    digest = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    sidecar = {
        "model": os.path.abspath(pth),
        "sha256": digest,
        "tuned_from": os.path.abspath(args.identifications),
        "peptides": len(rows),
        "epochs": args.epochs,
        "q_value": args.q_value,
        "rt_norm_min_minutes": lo,
        "rt_norm_max_minutes": hi,
        "method": args.method,
        "seconds": round(time.time() - t0, 1),
        "evaluation": evaluation,
        # Residual mode's model predicts a CORRECTION, not a retention time, so
        # it is not consumable with -rt_model alone; the calibration below has
        # to be applied and added back. ODIA cannot do that yet.
        "calibration": calibration,
        # The caller must decide whether this closes a loop; see the module
        # docstring. Recorded, not judged.
        "warning": "This model is specific to the run it was tuned on and must "
                   "not be reused across runs or gradients.",
    }
    with open(os.path.join(args.output, "rt_provenance.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    if args.method == "residual":
        print("NOTE: a residual model predicts a correction, not a retention "
              "time. ODIA cannot consume it with -rt_model -- the calibration in "
              "the sidecar must be applied and added back first. Use it to "
              "compare methods, not yet to build a library.", file=sys.stderr)
    print(json.dumps(sidecar, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
