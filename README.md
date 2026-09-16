# DIALibRefine

Make a predicted DIA spectral library right for one run, two ways:

- **`DIALibraryRefiner`** — replace predicted retention time and ion mobility
  with what the run actually measured, and delete the precursors it never saw
  (the peptide-centric reconstruction of Charkow *et al.*).
- **`DIALibTune`** — re-train AlphaPeptDeep's RT and CCS models on the run's
  identifications, in C++ with libtorch, and write them back into the stock
  ONNX files so that every predicted precursor gets the run's RT scale and
  mobility calibration. No Python.

The counterpart to [DIALibGen](https://github.com/okohlbacher/DIALibGen),
which predicts a library from a FASTA. This corrects one, and corrects the
models that predicted it.

> **Status: pre-release (0.1.0).** The config schema and the output contracts
> are not frozen. Everything below is measured; the numbers carry their run
> and date in [docs/results.md](docs/results.md).

## The method

`DIALibraryRefiner` implements the "peptide-centric library reconstruction"
of Charkow, Ghaznavi, Seale, Peng, Gingras & Röst, *Reference-Based Library
Construction Improves Performance in low-input diaPASEF Workflows*,
[bioRxiv 10.64898/2026.04.29.721088](https://www.biorxiv.org/content/10.64898/2026.04.29.721088v2).

Three operations, in this order:

1. **Filter** to precursors identified in the reference, at three q-value levels
   simultaneously (precursor, peptide, protein).
2. **Replace** retention time with the observed value.
3. **Replace** 1/K0 with the observed value (opt-in).

It is deliberately **not a fit**. Nothing is regressed onto anything; the
observed value is written through. There is therefore no generalisation — a
precursor absent from the reference gets no correction at all, which is why the
filter and the replacement always travel together.

**Filtering is the largest single component**, not a tidying step. In the paper,
library specificity accounts for 87% of reproducibility variance in OpenSWATH
and 64% in DIA-NN, and filtering *alone* beats their transfer-learning arm. The
mechanism they name is that OpenSWATH estimates the proportion of nulls in the
library when computing q-values, so a library that is >98% never-observed
hypotheses is being scored against a null it invented.

`DIALibTune` is the paper's *downstream* stage — transfer learning on the
run's identifications — for the precursors the run did not see. It keeps
AlphaPeptDeep's own fine-tuning recipe, adds protein-level held-out cohorts, a
measured stopping rule, and a refusal to write a model that did not beat the
stock one. [docs/fine-tuning.md](docs/fine-tuning.md) is the contract.

### What neither does

- **m/z is never touched**, and the paper does not touch it either. A
  precursor's m/z follows from its sequence, charge and modifications; what
  drifts is the instrument's mass scale, which is a property of the *run* and
  has no static-column representation in a library. A library whose m/z is wrong
  has wrong arithmetic, not miscalibration.
- **Fragment intensities are not replaced.** `-write_intensity` refuses rather
  than no-ops. The paper measures intensity replacement as a wash — *"RMSD in
  relative fragment ion intensity remain comparable between approaches"* across
  three separate figures — so it is the one component with no evidence behind it.
  The MS2 model is not tuned for the same reason.

## Building

Needs an installed OpenMS ≥ 3.5, Arrow/Parquet, ONNX Runtime and nlohmann/json;
`DIALibTune` additionally needs a CXX11-ABI libtorch. DIALibGen is fetched at
the pinned tag. The conda recipe CI uses, and every option, is in
[docs/install.md](docs/install.md).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DDLR_BUILD_FINETUNE=ON -DTorch_DIR=<libtorch>/share/cmake/Torch \
      -DDLR_PEPTDEEP_ONNX_DIR=<dir with peptdeep_{rt,ccs,ms2}_dynamic.onnx>
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Release bundles (Linux x64/arm64, macOS arm64/x64 — signed, notarized,
stapled `.dmg`) come from `.github/workflows/ci.yml`; see
[docs/release.md](docs/release.md).

Note what is **not** a dependency: mzPeak, SQLite, and the extraction stack.
Both tools read a library or a model and a *results table*, never the raw run.
That is what keeps them the same shape as DIALibGen rather than a search engine
with a different `main()`.

## Usage

```bash
DIALibraryRefiner -in predicted_library.parquet -ids report.parquet \
                  -out refined_library.parquet -out_report residuals.tsv -write_im

DIALibTune -in report.parquet -model_in models/peptdeep_rt_dynamic.onnx \
           -out tuned/peptdeep_rt_dynamic.onnx -head rt -filter:rt_max_minutes 30
DIALibTune -in report.parquet -model_in models/peptdeep_ccs_dynamic.onnx \
           -out tuned/peptdeep_ccs_dynamic.onnx -head ccs
```

`-ids` / `-in` is a DIA-NN `report.parquet` of one run. For the refiner, every
enabled q-value gate must find its column there, or the tool refuses — nothing
fails open; `-empirical_library` bypasses (and records) the gates a DIA-NN
`--gen-spec-lib` library does not carry. `-write_im` adds the mobility
replacement for charges ≥ 2 (z1 is censored at the ramp top); `-im_ramp_top`
declares the instrument's limit; `-no_filter` turns the filter off as a
declared arm. For the tuner, `-machine:device cuda:0` uses a GPU;
`-cohort:train_size` subsamples.

Every run writes a provenance sidecar — `<out>.refine.json` /
`<out>.tune.json` — with the recipe, input hashes, the reference run, every
rejection count and the measured residuals; a Parquet output carries the same
JSON in its schema metadata (`odia.config_json`).
[docs/quickstart.md](docs/quickstart.md) walks through both.

### Modification naming is canonicalised, and this is load-bearing

DIA-NN writes `C(UniMod:4)`. A library generated through OpenMS writes
`C(Carbamidomethyl)`. They are the same modification and a verbatim join matches
neither to the other — silently, with no error, leaving a result that looks like
a slightly worse search rather than a broken join.

Measured on the test case below: a verbatim join matched 33,749 of 37,193
reference precursors. The 3,444 it dropped are **exactly** the
cysteine-containing precursors — 9.26% of the reference, all of them.

## Measured on a real run

S08, Bruker timsTOF diaPASEF, human liver, carbamidomethylated. Library:
ODIA/AlphaPeptDeep prediction of the human proteome, 4,991,901 target precursors.
Reference: DIA-NN's `--gen-spec-lib --unimod4` empirical library for the same
run, 37,193 targets with measured RT and 1/K0.

**Refinement**

| | |
|---|---|
| reference precursors matched | **36,574 / 36,574 (100.0%)** |
| library precursors | 9,617,705 → **72,979** (36,574 targets + 36,405 decoys) — a **99.24% cut** |
| RT residual before refinement | mean 1.286, **sd 8.436**, p95 \|resid\| 17.52 iRT |
| 1/K0 residual before refinement | mean **−0.0204**, **sd 0.0287**, p95 \|resid\| 0.0551 |
| wall / peak RSS | 57 s / 11.8 GB |

The residuals are reported *before* the overwrite, because afterwards they are
zero by construction and say nothing. They are the measurement the refinement is
worth.

**Fine-tuning** (models tuned on DIA-NN's identifications of the run, 22k / 25k
training units, protein-level hold-out; error on precursors from proteins the
models never saw, in the units a library carries):

| | RT calibrated sd (min) | 1/K0 sd, z≥2 |
|---|---|---|
| stock AlphaPeptDeep | 0.989 | 0.0180 |
| **tuned, `DIALibTune`** (full pool, 100 epochs) | **0.318** | **0.0148** |
| DIA-NN's own post-run refit (in-sample) | 0.352 | 0.0148 |

A full-proteome library from the tuned models, searched with DIA-NN on the same
run, gave **+8% identifications at matched entrapment budget** over the stock
models (39,440 vs 35,805 at q ≤ 0.01; entrapment FDP measured and lower). It is
a **same-run** result; cross-run transfer has not been measured.

The C++ trainer reproduces the Python reference it was ported from — cohorts
and stock metrics to the printed digit, the 100-epoch result within replicate
noise — at 1.7× its speed per epoch on the same CPU threads, and is the only
thing a user needs to run. How accuracy and runtime depend on training-set size
and on the stopping rule, and the GPU numbers, are in
[docs/results.md](docs/results.md).

## Interpreting the output

**Both outputs are per-run objects.** A refined library's RT column holds the
reference run's observed retention times, not iRT — the column's meaning has
changed, and the tool says so on every run. A tuned model has the run's
gradient and the instrument's mobility calibration in its weights. Both are
correct for that run and for runs acquired the same way, and wrong elsewhere.

**A reconstructed library launders its own false positives.** The authors concede
this: *"our approach may transfer false positives… FDR estimation to be more
liberal"*. Any q-value headline measured with a reconstructed library is
optimistic by construction. Compare at a matched entrapment budget.

## Documentation

[install](docs/install.md) · [quickstart](docs/quickstart.md) ·
[fine-tuning](docs/fine-tuning.md) · [results](docs/results.md) ·
[conventions](docs/conventions.md) · [release](docs/release.md) ·
[FAQ](docs/faq.md) · [CHANGELOG](CHANGELOG.md) · [CONTRIBUTING](CONTRIBUTING.md)

## Licence

BSD-3-Clause. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
Cite with [CITATION.cff](CITATION.cff); the method is Charkow *et al.* and the
models are AlphaPeptDeep (Zeng *et al.*, 2022).
