# DIALibRefine

Refine a DIA spectral library against a reference run: replace predicted
retention time and ion mobility with what was actually measured, and delete the
precursors the reference never saw.

The counterpart to [DIALibGen](https://github.com/okohlbacher/DIALibGen), which
predicts a library from a FASTA. This one corrects one.

> **Status: pre-release (0.1.0).** The config schema and the output contract are
> not frozen.

## The method

This implements the "peptide-centric library reconstruction" of Charkow,
Ghaznavi, Seale, Peng, Gingras & Röst, *Reference-Based Library Construction
Improves Performance in low-input diaPASEF Workflows*,
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

### What it does not do

- **m/z is never touched**, and the paper does not touch it either. A
  precursor's m/z follows from its sequence, charge and modifications; what
  drifts is the instrument's mass scale, which is a property of the *run* and
  has no static-column representation in a library. A library whose m/z is wrong
  has wrong arithmetic, not miscalibration.
- **Fragment intensities are not replaced.** `-write_intensity` refuses rather
  than no-ops. The paper measures intensity replacement as a wash — *"RMSD in
  relative fragment ion intensity remain comparable between approaches"* across
  three separate figures — so it is the one component with no evidence behind it.
- **The C++ tool does not fine-tune models.** Transfer learning is a *downstream*
  stage in the paper, not an alternative: its training set is the reconstructed
  library. That stage is provided as scripts under `tools/` (see below), so torch
  never enters the tool.

## Building

Needs an installed OpenMS, Arrow/Parquet ≥ 23, and nlohmann/json. DIALibGen is
fetched automatically.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To build against a local DIALibGen checkout:
`-DFETCHCONTENT_SOURCE_DIR_DIALIBRARYGENERATOR=../DIALibGen`

Note what is **not** a dependency: mzPeak, SQLite, and the extraction stack.
This tool reads a library and a *results table*, never the raw run. That is what
keeps it the same shape as DIALibGen rather than a search engine with a
different `main()`.

## Usage

```bash
DIALibraryRefiner \
  -in  predicted_library.parquet \
  -ids reference_run_report.parquet \
  -out refined_library.parquet \
  -out_report residuals.tsv
```

`-ids` accepts a DIA-NN `report.parquet` or a DIA-NN empirical library
(`--gen-spec-lib` output). `-write_im` adds the mobility replacement;
`-no_filter` turns the filter off as a declared arm.

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

Both independently reproduce numbers this project had measured by other means:
an RT residual sd of 8.42 and a 1/K0 residual sd of 0.02848 for the same library
against the same reference. The 1/K0 mean of −0.0204 against a mean 1/K0 near
1.0 is the ~2% low bias in predicted mobility, recovered here as a side effect.

## Interpreting the output

**The refined library is a per-run object.** Its RT column holds the reference
run's observed retention times, not iRT — the column's meaning has changed, and
the tool says so on every run. It is correct for that run and for runs on the
same gradient, and wrong elsewhere. Cross-run transfer of a per-run RT model has
not been cleanly measured in this project; treat the output as per-run until it is.

**A reconstructed library launders its own false positives.** The authors concede
this: *"our approach may transfer false positives… FDR estimation to be more
liberal"*. Any q-value headline measured with a reconstructed library is
optimistic by construction. Compare at a matched entrapment budget.

## Fine-tuning (`tools/`)

The paper puts transfer learning *downstream* of reconstruction, trained on the
reconstructed identifications. That stage lives here as scripts — torch stays
out of the C++ tool:

| script | does |
|---|---|
| `tools/finetune_rt.py` | fine-tune the AlphaPeptDeep RT head on a run's identifications; `--holdout protein` (a sequence split leaks co-eluting siblings) |
| `tools/finetune_ccs.py` | the same for the CCS head — the first time it has been done in this project; guards peptdeep's silent no-op and excludes censored z1 |
| `tools/export_finetuned.sh` | any head → the ONNX triplet DIALibGen consumes, with a manifest naming which checkpoint each file came from |

Then point DIALibGen's `rt_model` / `ccs_model` at the exports and regenerate.

**Measured on S08** (Bruker timsTOF diaPASEF, carbamidomethylated; models
tuned on DIA-NN's own identifications of that run, 26k peptides, protein-level
holdout). Library accuracy on precursors from proteins the models never saw:

| | RT sd, monotone (min) | 1/K0 sd, z≥2 |
|---|---|---|
| DIA-NN raw library | 0.5065 | 0.01602 |
| ODIA stock | 0.6875 | 0.01796 |
| **ODIA fine-tuned** | **0.3035** | **0.01475** |
| DIA-NN post-run-refit (in-sample) | 0.3524 | 0.01480 |

And in a DIA-NN search of the same run, full-proteome libraries from the same
FASTA and config, differing only in the two ONNX files:

| | q≤0.01 | protein groups | entrapment FDP | at matched entrapment budget |
|---|---|---|---|---|
| DIA-NN library-free | 37,334 | 5,033 | 2.67% | — |
| ODIA library, stock models | 35,805 | 4,907 | 2.34% | parity (±1%) |
| **ODIA library, fine-tuned** | **39,440** | **5,228** | **2.45%** | **+8%** |

The gain is not error inflation — entrapment FDP is measured and lower — and it
is not a reordering: 5,932 precursors were found only with the fine-tuned
library. **It is a same-run result.** The models were tuned on the run they are
searching; cross-run transfer has not been measured. Treat the +8% as a per-run
property until it has.

## Licence

BSD-3-Clause. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
