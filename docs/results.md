# Results

Every number here is dated and names its run, node and artefact. Nothing is
extrapolated. The run is **S08** throughout: Bruker timsTOF diaPASEF, human
liver, carbamidomethylated, 30-min gradient; reference = DIA-NN 2.x
library-free search of that run (`A_libfree.parquet`, 37,334 precursors at
q ≤ 0.01), models = stock AlphaPeptDeep exports (`peptdeep_rt_dynamic.onnx`
sha256 3f3b847e…, `peptdeep_ccs_dynamic.onnx` 41816f32…).

## Refinement (`DIALibRefine`, 2026-09-14)

Library: ODIA/AlphaPeptDeep prediction of the human proteome, 4,991,901
target precursors. Reference: DIA-NN's `--gen-spec-lib --unimod4` empirical
library for the run, 37,193 targets with measured RT and 1/K0.

| | |
|---|---|
| reference precursors matched | **36,574 / 36,574 (100.0 %)** |
| library precursors | 9,617,705 → **72,979** (36,574 targets + 36,405 decoys) — a 99.24 % cut |
| RT residual before refinement | mean 1.286, **sd 8.436**, p95 \|resid\| 17.52 iRT |
| 1/K0 residual before refinement | mean −0.0204, **sd 0.0287**, p95 \|resid\| 0.0551 |
| wall / peak RSS | 57 s / 11.8 GB |

The 1/K0 mean of −0.0204 against a mean 1/K0 near 1.0 is the ~2 % low bias in
predicted mobility this project had measured by other means.

## Library-level effect (DIA-NN arms, 2026-09-15)

Full-proteome libraries from the same FASTA and DIALibGen config, differing
only in the two ONNX files, searched with DIA-NN on the same run (same-run
result; cross-run transfer not measured):

| | q ≤ 0.01 | protein groups | entrapment FDP | at matched entrapment budget |
|---|---|---|---|---|
| DIA-NN library-free | 37,334 | 5,033 | 2.67 % | — |
| ODIA library, stock models | 35,805 | 4,907 | 2.34 % | parity (±1 %) |
| **ODIA library, fine-tuned models** | **39,440** | **5,228** | **2.45 %** | **+8 %** |

Not error inflation (entrapment FDP measured and lower) and not a reordering:
5,932 precursors were found only with the fine-tuned library.

## Held-out accuracy of the tuned models (2026-09-17)

Protein-level TEST cohort (`crc32(Protein.Group) % 5 == 0`), never used for
training or selection; error of the deployed prediction (minutes after
undoing `rt_norm`; 1/K0 through Mason–Schamp) against the observation.
Full pool, 100 epochs.

| | RT calibrated sd (min) | RT sd | 1/K0 sd, z ≥ 2 |
|---|---|---|---|
| stock AlphaPeptDeep | 0.989 | 2.635 | 0.0180 |
| tuned, Python reference (`tools/finetune.py`, sweep `n0_h100`) | 0.3176 | 0.3183 | 0.0148 |
| tuned, **`DIALibTune`** (H100, seeds 1/2/3) | **0.3175 / 0.3176 / 0.3176** | 0.3176 | see GPU table |
| DIA-NN's own post-run refit (in-sample) | 0.352 | — | 0.0148 |

The stock RT model's raw sd (2.6 min) is mostly scale — the run's gradient is
not in the model — which the affine calibration removes (0.99); fine-tuning
removes the rest of it and two thirds of the remaining scatter.

## C++ vs Python: acceptance

Same report, same filters, same cohorts. Cohort counts and stock metrics are
identical to the printed digit (RT: 32,576 units, 143 rejected for RT spread,
TEST/VAL/pool 6,712/3,812/22,052, stock TEST 2.6346/0.98927; CCS: 36,513
units, 821 charge-1 rejections, 7,433/4,272/24,808, 0.01796/0.01710).

Replica of the sweep's 2,000-unit, 100-epoch RT run, both on 4 CPU threads
on the same node (dax, 2026-09-17):

| | best VAL cal. sd | @ epoch | TEST cal. sd | updates | train s | eval s |
|---|---|---|---|---|---|---|
| Python (`finetune.py`) | 0.3932 | 91 | 0.3873 | 2,400 | 167 | 83 |
| C++ (`DIALibTune`) | 0.3914 | 91 | 0.3879 | 2,400 | 98 | 53 |

The subsample draws differ (numpy vs `std::mt19937`), so this is two draws of
the same pool; the replicate sd of the held-out metric is 0.004 min. The C++
is 1.7× faster per epoch at equal threads, from the same recipe.

## Convergence thresholds

One continuous 100-epoch trajectory per training size, per-epoch validation,
patience off; every stopping rule replayed offline on the stored trajectory
(`tools/sweep.sh`, `tools/sweep_report.py`; `/scratch/kohlbach/odia/ft2/sweep`
on dax, 4 threads per job, 2026-09-16/17). Rule: progress = beat the anchor
by *rel*; stop after *patience* non-progress epochs, never before epoch 20.
"Regret" = VAL metric at the stop vs the best the trajectory ever reached.

### RT (VAL calibrated sd, minutes)

| units | run to 100: best VAL / TEST / train s | rel 0.5 % pat 10 | rel 1 % pat 10 | rel 2 % pat 10 | rel 5 % pat 3 |
|---|---|---|---|---|---|
| 500 | 0.4795 / 0.4622 / 76 | 68 ep, +0.5 % | 63, +0.6 % | 51, +1.9 % | 29, +9.7 % |
| 1,000 | 0.4284 / 0.4120 / 106 | 56, +0.9 % | 56, +0.9 % | 48, +1.0 % | 29, +6.7 % |
| 2,000 | 0.3932 / 0.3873 / 167 | 58, +0.5 % | 51, +1.0 % | 42, +1.7 % | 26, +6.7 % |
| 4,000 | 0.3722 / 0.3692 / 319 | 58, +0.8 % | 46, +1.4 % | 46, +1.4 % | 22, +9.9 % |
| 8,000 | 0.3577 / 0.3547 / 576 | 72, +0.2 % | 56, +1.1 % | 48, +1.6 % | 23, +9.9 % |
| 16,000 | 0.3310 / 0.3290 / 1,036 | 78, +0.3 % | 58, +1.1 % | 47, +2.1 % | 21, +12.2 % |
| 22,052 (full) | 0.3216 / 0.3176 / 1,330 | 73, +0.3 % | 64, +0.6 % | 48, +2.3 % | 20, +12.9 % |

### CCS (VAL calibrated sd, 1/K0; stock TEST 0.0180)

| units | run to 100: best VAL / TEST / train s | rel 0.5 % pat 10 | rel 1 % pat 10 | rel 2 % pat 10 | rel 5 % pat 3 |
|---|---|---|---|---|---|
| 500 | 0.0160 / 0.0162 / 80 | 24 ep, +0.9 % | 28, +0.3 % | 21, +0.9 % | 20, +0.9 % |
| 1,000 | 0.0158 / 0.0161 / 107 | 36, +0.1 % | 35, +0.1 % | 30, +0.2 % | 20, +1.4 % |
| 2,000 | 0.0157 / 0.0160 / 168 | 34, +1.4 % | 34, +1.4 % | 30, +1.4 % | 20, +2.3 % |
| 4,000 | 0.0156 / 0.0158 / 297 | 43, +0.3 % | 28, +1.9 % | 28, +1.9 % | 20, +2.4 % |
| 8,000 | 0.0155 / 0.0156 / 560 | 53, +0.0 % | 31, +1.4 % | 24, +2.2 % | 20, +2.6 % |
| 16,000 | 0.0150 / 0.0151 / 975 | 51, +0.0 % | 34, +0.9 % | 34, +0.9 % | 20, +2.8 % |
| 24,808 (full) | 0.0148 / 0.0148 / 1,390 | 57, +0.3 % | 34, +1.6 % | 33, +1.6 % | 20, +3.3 % |

**Reading.** The default (rel 0.5 %, patience 10) stops RT at 56–78 epochs
within 0.9 % of the best and CCS at 24–57 within 1.4 %. Coarser rules buy
little: rel 5 %/patience 3 stops RT at epoch 20–29 and leaves 6–13 % on the
table. CCS converges early at every size (best epoch 32–82, the curve is flat
after ~30). Updates per 100 epochs: 2,300–2,400 for ≤ 8,000 units, 3,200 at
16,000, 3,700–3,900 at the full pool — cost is steps, and steps barely depend
on size below 8,000 units because batches are per peptide length.

### The horizon confound

A run whose cosine anneals at its own (short) horizon vs the 100-horizon
trajectory cut at the same epoch — measured, not assumed:

| units | horizon | real run, best VAL | 100-horizon cut at h | real run TEST |
|---|---|---|---|---|
| 2,000 (RT) | 20 | 0.549 | 0.458 | 0.533 |
| 2,000 (RT) | 40 | 0.421 | 0.401 | 0.415 |
| full (RT) | 20 | 0.385 | 0.382 | — |
| full (RT) | 40 | 0.349 | 0.345 | — |
| 2,000 (CCS) | 20 / 40 | 0.0162 / 0.0160 | 0.0160 / 0.0157 | 0.0164 / 0.0163 |

Shortening the horizon is worse than stopping a long one early, so the
horizon stays at 100 and patience decides.

## Subsampling

From the tables above, TEST after 100 epochs vs the full pool: 500 units
recover 78 % of the RT gain and 56 % of the CCS gain; 2,000 units 90 % / 63 %;
8,000 units 95 % / 75 %; 16,000 units 98 % / 91 %. There is no plateau; the
default is the full pool. On a CPU the training time scales with the rows
(76 s → 1,330 s per 100 epochs at 4 threads); on the GPU below it does not
matter.

## GPU

Node `data` (2× H100 PCIe, driver 580.173, one card shared with another
user's 69 GB job at 0–2 % utilisation), `DIALibTune` built against libtorch
2.14.0+cu130 with cuDNN 9.26 ([install.md](install.md#cuda)), full pool,
2026-09-17. `nvidia-smi` sampled every 5 s beside every run; the card used
sat at 66–68 % mean utilisation (79–85 % peak) and +2.7 GB memory during the
RT runs, so the wall clock is a measurement of the tool, not of contention.

| run | epochs | updates | train s | wall s | GPU | TEST |
|---|---|---|---|---|---|---|
| RT, full horizon, seed 1 | 100 | 3,700 | **27.8** | 31.7 | cuDNN | 0.3175 min |
| RT, full horizon, seed 2 | 100 | 3,700 | 28.0 | 32.0 | cuDNN | 0.3176 |
| RT, full horizon, seed 3 | 100 | 3,700 | 28.3 | 32.3 | cuDNN | 0.3176 |
| CCS, full horizon, seed 1 | 100 | 3,900 | 30.0 | 34.8 | cuDNN | 0.01481 (1/K0 sd) |
| RT, full horizon, no cuDNN (`-machine:no_cudnn`) | 100 | 3,700 | 254.7 | 276.2 | native, 88 % util | 0.3176 |
| RT, **default stopping rule** (stopped at 69, best 69) | 69 | 2,553 | **19.3** | 23.2 | cuDNN | 0.3195 |
| CCS, **default stopping rule** (stopped at 47, best 46) | 47 | 1,833 | **14.2** | 17.3 | cuDNN | 0.01489 |
| RT, full horizon, **CPU** (same node, 8 threads) | 100 | 3,700 | 631.4 | 667.6 | — | 0.3181 |
| RT, default rule, **final code** (8ad69d9) | 79 | 2,923 | 20.6 | 24.8 | cuDNN | 0.3185 |
| CCS, default rule, **final code** (8ad69d9) | 44 | 1,716 | 12.1 | 15.3 | cuDNN | 0.01486 |
| RT, full horizon, Python, CPU (dax, 4 threads) | 100 | 3,700 | 1,330 | ~1,400 | — | 0.3176 |

7.5 ms per optimizer step on the H100 with cuDNN (0.28 s per epoch of 37
steps); libtorch's native LSTM kernels are 9× slower (69 ms per step) at
higher utilisation, so the cuDNN library set is worth installing
([install.md](install.md#cuda)). The per-epoch validation adds 3.2 s over a
100-epoch run. The default stopping rule ended RT at epoch 69 (TEST 0.3195
vs 0.3175 for the full horizon, +0.6 %) and CCS at 47 (0.01489 vs 0.01481) —
the regret the sweep predicted. Three seeds agree on TEST
to 0.0001 min — with the full pool there is no subsample draw, only batch
order and dropout. The wall-clock overhead outside training (reading the
report, encoding 32,576 peptides, stock evaluation, write-back) is ~4 s.

**What this means for cost.** The same binary, same node, same run: the
H100 trains the full-pool 100-epoch RT model **22.7× faster** than 8 CPU
threads (27.8 s vs 631 s), and 48× faster than the Python reference on 4
threads (1,330 s, on dax). A full-pool fine-tune of both heads with the
default stopping rule is ~40 s of H100 wall time including I/O; on a CPU it
is ~10–12 minutes per head. The constraint that started this — keep
convergence slow, do not burn GPU time — is met by the recipe (lr 1e-4,
cosine to 100) and by the patience rule, not by cutting the horizon; at
7.5 ms a step there is no GPU time worth saving by stopping earlier than
the rule does.

The rows marked *final code* were run after the second review pass (device-
side gradient clipping, anchor at the first checkpoint); the other GPU rows
ran the pre-review-2 binary (commit d8d8d0a) with the identical recipe.

## Provenance of these numbers

- sweep: `/scratch/kohlbach/odia/ft2/sweep/{rt,ccs}/n*_h*/` (dax), one
  `trajectory.tsv` and `*_provenance.json` per run; `tools/sweep_report.py`
  reproduces both tables.
- C++ replica: session scratchpad `smoke/rt_n2000_h100/` (dax).
- GPU: `shared/ft/gpu_bench/bench/<run>/` — `log.txt`, `nvidia-smi.csv`,
  `*.tune.json`, `*.trajectory.tsv`; the scripts that produced them are in the
  same directory.
- DIA-NN arms: `shared/refined/diann_arms/{A..E}*` and `shared/refined/libgen/`.
