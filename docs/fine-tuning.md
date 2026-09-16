# Fine-tuning the RT and CCS models on one run

`DIALibTune` re-trains AlphaPeptDeep's retention-time and CCS models on a
single run's DIA-NN identifications, in C++ with libtorch, and writes the
result back into the stock ONNX file so that
[DIALibGen](https://github.com/okohlbacher/DIALibGen) consumes it unchanged.
No Python is involved at run time.

This page is the contract: what goes in, what the recipe is, how the tool
decides it has converged, how much data it needs, and what it refuses to do.
The numbers behind every default are in [results.md](results.md).

## What it is for

The Charkow–Röst reconstruction ([README](../README.md#the-method)) corrects
the precursors a reference run *saw*. A fine-tuned model corrects the ones it
did not: it learns the run's RT scale and the instrument's mobility
calibration from the identified precursors and applies them to the whole
predicted library. On S08 a library generated from the tuned models gave DIA-NN
**+8% identifications at matched entrapment budget** over the stock models
([results.md](results.md#library-level-effect)).

**The tuned model is a per-run object.** The run's gradient and the
instrument's mobility calibration are baked into it. It is right for that run
and for runs acquired the same way, and wrong elsewhere; cross-run transfer
has not been measured in this project.

## Inputs

| input | what |
|---|---|
| `-in` | a DIA-NN `report.parquet` of **one** run (a multi-run report is refused) |
| `-model_in` | the stock `peptdeep_rt_dynamic.onnx` or `peptdeep_ccs_dynamic.onnx` (DIALibGen ships them) |
| `-head rt\|ccs` | which of the two the file is |
| `-out` | the tuned ONNX; `<out>.tune.json` (provenance, metrics) and `<out>.trajectory.tsv` (per epoch) are written beside it |

### Which rows become training data

The observation filter is the same as the Python reference tool's and is
recorded, count by count, in the provenance:

- `Q.Value` finite and ≤ `filter:q_value` (default 0.01);
- not a decoy (`Decoy` column, if present, must be 0);
- charge integral, 1–8; **for CCS** additionally ≥ `filter:min_charge`
  (default 2; charge 1 is censored at the mobility ramp top on timsTOF and
  needs `-filter:allow_z1` to be admitted);
- finite `RT` ≥ 0, finite `Precursor.Mz` > 0, and for CCS finite `IM` > 0;
- a non-empty `Protein.Group`.

Rows are then collapsed into **units**:

- **RT**: one unit per modified sequence, all charge states pooled, target =
  median RT. A sequence whose charge states disagree by more than
  `filter:rt_spread_max` (0.2 min) is rejected.
- **CCS**: one unit per (modified sequence, charge), target = median 1/K0,
  converted to CCS through Mason–Schamp with the constants DIALibGen uses
  (`ODIA::ccsFromMobility`) at the precursor's m/z and charge.

A unit whose observations disagree on the protein group is rejected — the
group decides the cohort, and a first-row rule would let the row order decide
it.

What the model predicts is `rt_norm` = RT / `filter:rt_max_minutes` (default:
the run's maximum observed RT) and CCS in Å². **Every reported number is in the
deployed unit** — minutes, and 1/K0 — after undoing those transforms, because
that is what a library carries.

### Cohorts

Three protein-level cohorts, frozen *before* any subsampling, so the held-out
sets are identical across every training size:

| cohort | rule | used for |
|---|---|---|
| TEST | `crc32(Protein.Group) % 5 == 0` | reported once at the end; never touched otherwise |
| VAL | `crc32("val:" + Protein.Group) % 7 == 0` of the rest | checkpoint selection and stopping |
| pool | everything else | training: the full pool, or a seeded random prefix of it (`cohort:train_size` / `cohort:train_frac`) |

Protein-level, not sequence-level: a sequence split leaks co-eluting siblings
of the same protein into the held-out set and flatters every number.

Both held-out cohorts must have at least 100 units or the tool refuses — below
that, the affine calibration behind the selection metric fits noise.

`-cohort:no_inner_val` dissolves VAL into the pool and selects checkpoints on
TEST. That is the reference tool's behaviour too, and it makes TEST an
optimistic number; the provenance records `val_is_test: true`.

## The recipe

Unchanged from AlphaPeptDeep's own transfer-learning defaults, and from the
Python tool this replaces:

| | |
|---|---|
| loss | L1 (mean) |
| optimizer | Adam, lr 1e-4, β (0.9, 0.999), ε 1e-8, no weight decay |
| gradient clipping | global norm 1.0 |
| batches | ≤ 1024, **within one peptide length**, never padded (padding changes the attention softmax) |
| schedule | per epoch: linear warmup over `train:warmup` (10) epochs, then cosine to `train:epochs` (100) |
| dropout | 0.1 after the attention sum, training only |
| frozen | the LSTM initial states `h0`/`c0`, as in peptdeep |

The learning rate is deliberately slow and the horizon deliberately long. A
run that anneals its cosine at epoch 20 ends *worse* than a 100-horizon run
stopped at epoch 20 (measured, both heads, [results.md](results.md#the-horizon-confound));
so the horizon stays at 100 and convergence is decided by patience, not by
shortening the schedule.

## Convergence and stopping

Every `stop:eval_every` (1) epochs the model is evaluated on VAL in deployed
units. The selection metric (`stop:select`) is the **calibrated sd**: the sd
of the residual after fitting observed = a·predicted + b. It is what a
library consumer sees, because DIA-NN and ODIA both refit an affine RT/IM
map per run before using the library; `rmse` is available for the raw error.

The stopping rule is the anchor-patience rule the sweep replayed offline:

- a checkpoint is *progress* if it beats the anchor by
  `max(stop:abs_tol, stop:rel_tol × anchor)` (defaults 0, 0.5%); it then
  becomes the anchor;
- after `stop:patience` (10) epochs without progress the run stops — never
  before `max(stop:min_epochs, train:warmup)` (20, 10);
- the checkpoint with the best VAL metric is restored and written;
- `stop:max_seconds` is a wall-clock budget on training; the state it stops
  at is evaluated, so no run ends without a checkpoint having been looked at.

**Selection starts from the stock model.** If no checkpoint beats the stock
model's VAL metric, nothing is written to `-out`, the provenance still is
(`exported: false`), and the tool exits non-zero. A "tuned" file that is
worse than stock does not get to exist.

### Measured thresholds

From the size × stopping-rule sweep on S08 (one 100-epoch trajectory per size,
every rule replayed on it; [results.md](results.md#convergence-thresholds)):

| rule | RT: epochs run / regret vs the run's best | CCS: epochs / regret |
|---|---|---|
| rel 0.5 % · patience 10 (**default**) | 56–78 / ≤ 0.9 % | 24–57 / ≤ 1.4 % |
| rel 1 % · patience 10 | 46–64 / ≤ 1.4 % | 28–35 / ≤ 1.9 % |
| rel 2 % · patience 10 | 40–48 / ≤ 2.3 % | 20–34 / ≤ 2.2 % |
| rel 5 % · patience 3 | 20–29 / 6–13 % | 20 / 1–3 % |

"Regret" is how far the VAL metric at the stop is above the best the same
trajectory ever reached. The default buys < 1 % regret for roughly 60 % of a
full 100-epoch run on RT, and stops CCS — which converges in 30–50 epochs at
every size — well before the horizon.

## How much data: subsampling

Training cost is optimizer **steps**, not peptides: batches are per peptide
length, so a 500-unit subsample still takes ~23 steps per epoch against 38
for the full 22k-unit pool. On a GPU, where a 1024-row batch costs about what
a 64-row one does, a subsample therefore saves little time. It matters on a
CPU, where an epoch's cost scales with the rows.

Measured on S08, TEST metric after a full 100-epoch run (stock RT 0.989 min,
stock 1/K0 0.0180):

| training units | RT TEST calibrated sd (min) | 1/K0 TEST sd | CPU train s (4 threads, 100 ep) |
|---|---|---|---|
| 500 | 0.462 | 0.0162 | 76 / 80 |
| 1 000 | 0.412 | 0.0161 | 106 / 107 |
| 2 000 | 0.387 | 0.0160 | 167 / 168 |
| 4 000 | 0.369 | 0.0158 | 319 / 297 |
| 8 000 | 0.355 | 0.0156 | 576 / 560 |
| 16 000 | 0.329 | 0.0151 | 1 036 / 975 |
| full pool (22 052 / 24 808) | **0.318** | **0.0148** | 1 330 / 1 390 |

RT gains ~0.02 min per doubling and is still improving at the full pool; CCS
gains are small in absolute terms (0.0180 → 0.0148) and also grow with size.
There is no plateau to stop at — **the default is the full pool.** Subsample
(`-cohort:train_size 8000` gets 95 % of the RT gain at 43 % of the CPU time)
only when the machine, not the result, is the constraint.

## Running it

```bash
DIALibTune -in report.parquet -model_in models/peptdeep_rt_dynamic.onnx \
           -out tuned/peptdeep_rt_dynamic.onnx -head rt -filter:rt_max_minutes 30
DIALibTune -in report.parquet -model_in models/peptdeep_ccs_dynamic.onnx \
           -out tuned/peptdeep_ccs_dynamic.onnx -head ccs
cp models/peptdeep_ms2_dynamic.onnx tuned/          # MS2 is not tuned
DIALIBGEN_MODEL_DIR=tuned DIALibGen -in proteome.fasta -out library.parquet
```

`-filter:rt_max_minutes` should be the gradient length; left at 0 it is the
run's maximum observed RT, which is fine for the run itself and makes the
scale part of the provenance either way.

`-machine:device cuda:0` runs on a GPU (a CUDA libtorch build, see
[install.md](install.md#cuda)); on a CPU `-machine:threads 4` was the fastest
setting measured for this model — more threads thrash.

## What it refuses

- a multi-run report; a report missing a required column;
- `-out` equal to `-model_in`;
- CCS training on charge 1 without `-filter:allow_z1`;
- an explicit `rt_max_minutes` below the report's maximum RT;
- fewer than 100 VAL or TEST units;
- a run in which no checkpoint beat the stock model.

## Provenance

`<out>.tune.json` records the tool, the libtorch version, the recipe and
stopping rule as configured, the filter and every rejection count, the
cohort sizes and rule, the input model's SHA-256 and the output's, the course
of training (epochs, best epoch, updates, seconds, stop reason), and the
stock and tuned metrics on VAL and TEST — n, rmse, sd, mean error, p95,
calibrated sd with the fitted slope and intercept, and per-charge sd for CCS.
`<out>.trajectory.tsv` has one row per epoch (lr, updates, training loss,
every VAL metric, cumulative seconds, whether it was the best, the stop
reason).

The ONNX itself is the stock file with 21 initializers' bytes replaced — same
graph, same opset, same size — so anything that could read the stock model
reads the tuned one.

## Where the Python tools stand

`tools/finetune.py` is the reference implementation this was ported from and
tested against (same filters, cohorts, recipe and stopping rule; on S08 the
C++ reproduces its cohort counts and stock metrics to the printed digit and
its 100-epoch result within replicate noise). `tools/sweep.sh` and
`tools/sweep_report.py` produced the tables above. They need the `peptdeep`
Python environment and are kept for reproducing the measurements, not for
use.
