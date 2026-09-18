# Changelog

## [0.2.0] — 2026-09-18

**`DIALibTune` is gone.** Both stages are now one executable that always ends
with a library: `DIALibRefine` reconstructs, and `-tune` additionally fine-tunes
the RT and CCS models and re-predicts the whole library through them before the
reconstruction is applied. A tuned model is an intermediate, not a deliverable.

### Changed — breaking
- `DIALibTune` is removed. Its `-in report.parquet -model_in <onnx> -out <onnx>
  -head <h>` becomes `DIALibRefine -in <library> -ids <report> -out <library>
  -tune -tune_models <dir> [-tune_heads <h>]`, and the tuned ONNX files are kept
  only if `-tune_out_models <dir>` asks for them.
- The library is read from `-in` and re-predicted in place. The old route —
  tune, regenerate the library from the FASTA with `DIALIBGEN_MODEL_DIR`, then
  refine — needed three tools and a full re-digest to change two columns.
  `-tune_out_models` still produces a directory that works as a
  `DIALIBGEN_MODEL_DIR`, so that route remains available where it is wanted.
- The tuner's stage options are prefixed `tune_` (`-tune_models`,
  `-tune_heads`, `-tune_out_models`, `-tune_predict_gpu`,
  `-tune_predict_sessions`, `-tune_keep_free_cysteine_offset`). They could not
  keep a `tune:` prefix: OpenMS's `Param` will not accept `tune` as both a flag
  and a subsection, and the tool exits with `the element 'tune' (TOPP
  subsection not registered)`. The training recipe keeps its one-level
  `filter:`, `cohort:`, `train:`, `stop:` and `machine:` subsections unchanged.
- `test/standalone_test.sh` takes one tool path, not two.

### Fixed
- **The re-prediction count was inverted.** `predictRetentionTimes` and
  `predictCollisionCrossSections` return the number they could *not* predict;
  they were being reported as the number predicted, so a completely successful
  re-prediction printed `re-predicted 0`. Nothing caught it: `tune_e2e` asserted
  that the values changed, never that the count was non-zero. It does now.
- The re-predicted iRT is scaled back to minutes by `filter:rt_max_minutes`.
  The model emits `RT / rt_max_minutes`; leaving it normalised put the whole
  library on a 0–1 scale that nothing downstream expects.
- `-tune` without `-tune_models` named `-tune:models`, an option that no longer
  exists.

### Added
- `tune_e2e` asserts a non-zero re-prediction count per head, and its timeout is
  2400 s (the full end-to-end run measured 960 s on an H100 node, above the old
  900 s limit).
- `test/synth_report.py --library` writes a matching DIA-NN TSV library, so the
  end-to-end test refines a library that is larger than its reference (584
  precursors against 200 identified) — the case the merge exists for.

## [0.1.0] — 2026-09-18

First release. Two tools: peptide-centric library reconstruction (filter +
observed RT + observed 1/K0) against a reference run's identifications, and
fine-tuning of the AlphaPeptDeep RT and CCS models on those identifications.

### Added
- `DIALibRefine` TOPP tool and the `odia_refine` library.
- Canonical modification naming, so `C(UniMod:4)` and `C(Carbamidomethyl)` join.
  Measured worth on a carbamidomethylated timsTOF run: 3,444 precursors, which
  is exactly the cysteine-containing population.
- Per-axis residual reporting, measured BEFORE the overwrite.
- Refusal of `-write_intensity`, which the source paper measures as a wash.
- `-empirical_library` for DIA-NN `--gen-spec-lib` references (gates without a
  column are bypassed and each bypass recorded); `-im_ramp_top` / `-im_ramp_margin`
  for censored charge-1 mobilities; `-config` / `-write_config` with validation.
- Provenance: `<out>.refine.json` and the same JSON in the Parquet metadata
  (`odia.config_json`) — recipe, input hashes, reference run, rejection counts,
  residuals.
- **`DIALibTune`** and the `odia_tune` library: the RT and CCS models transcribed
  to libtorch, weights loaded from and written back into the stock ONNX bytes
  (no ONNX/protobuf library; byte-identical round trip; LSTM gate permutation
  and MatMul transposes handled by graph position). Same filters, protein-level
  cohorts, recipe and anchor-patience stopping rule as the Python reference;
  selection starts from the stock model and a model that did not beat it is not
  written. Provenance sidecar `<out>.tune.json` and a per-epoch trajectory.
  Verified against ONNX Runtime (RT 8.9e-8, CCS 1.2e-4) and against the Python
  tool on S08 (cohorts and stock metrics identical; 100-epoch result within
  replicate noise; 1.7× faster per epoch on the same CPU threads).
- Optional direct linking of an unpacked libtorch zip (`DLR_LIBTORCH_DIR`) for
  CUDA builds; `-machine:device cuda:N`; `-machine:no_cudnn`.
- Both tools report their own version, keep OpenMS's update check off and
  register with ToolHandler so `-write_ctd` works.
- Tests: join key; parity + round trip; standalone gates (bare environment);
  a synthetic single-run report and a bounded end-to-end fine-tune of both heads.
- CI (`.github/workflows/ci.yml`): Linux x64/arm64 and macOS arm64/x64 with
  OpenMS 3.5.0 and libtorch 2.10.0 from conda, the stock models cached by
  SHA-256, install + standalone gate, portable bundles, macOS Developer ID
  signing, notarization and a stapled `.dmg`, release asset assertion.
- Documentation tree under `docs/`, `CITATION.cff`, `CONTRIBUTING.md`.
- `tools/finetune.py` (the Python reference), `tools/sweep.sh` and
  `tools/sweep_report.py` (the size × stopping-rule sweep), `tools/export_finetuned.sh`.

### Measured (S08, docs/results.md)
- Held-out RT 0.989 → 0.318 min, 1/K0 0.0180 → 0.0148 (full pool, 100 epochs);
  DIA-NN's own in-sample refit: 0.352 / 0.0148. A full-proteome library from
  the tuned models gives DIA-NN +8% at matched entrapment budget on the same run.
- Convergence: the default rule (rel 0.5 %, patience 10, min 20 epochs, horizon
  100) stops RT at epoch 56–78 within 0.9 % of the trajectory's best and CCS at
  24–57 within 1.4 %.

### Known limitations
- Fragment intensity replacement is not implemented; the MS2 model is not tuned.
- Outputs are per-run objects; nothing prevents them being copied elsewhere.
- The tuned ONNX carries no embedded provenance (the sidecar does), so that
  every consumer of the stock file reads it unchanged.
- No Windows build; no released CUDA build.
- Cross-run transfer of tuned models is unmeasured.
