# Quickstart

Two tools, one run. Both take a DIA-NN `report.parquet` of the run as the
reference.

## 1. Refine a library against the run

```bash
DIALibraryRefiner -in predicted_library.parquet -ids report.parquet \
                  -out refined_library.parquet -out_report residuals.tsv -write_im
```

Keeps only the precursors the run identified (q ≤ 0.01 at precursor, peptide
and protein level), writes the observed RT — and with `-write_im` the observed
1/K0 for charge ≥ 2 — through, and reports the pre-overwrite residuals.
`refined_library.parquet.refine.json` records everything that happened.

If the reference is a DIA-NN empirical library (`--gen-spec-lib` output),
add `-empirical_library`: it carries no `Global.Q.Value`, so that gate is
bypassed and the bypass recorded.

## 2. Fine-tune the models on the run, and regenerate

```bash
DIALibTune -in report.parquet -model_in models/peptdeep_rt_dynamic.onnx  -out tuned/peptdeep_rt_dynamic.onnx  -head rt -filter:rt_max_minutes 30
DIALibTune -in report.parquet -model_in models/peptdeep_ccs_dynamic.onnx -out tuned/peptdeep_ccs_dynamic.onnx -head ccs
cp models/peptdeep_ms2_dynamic.onnx tuned/
DIALIBGEN_MODEL_DIR=tuned DIALibGen -in proteome.fasta -out library.parquet
```

Each `DIALibTune` run prints the stock and tuned held-out error in minutes /
1/K0, writes the tuned ONNX in place of the stock one's bytes, and leaves
`<out>.tune.json` beside it. It refuses to write a model that did not beat the
stock one on validation.

On a GPU add `-machine:device cuda:0`; on a CPU leave `-machine:threads` at 4.

## 3. Read the numbers

- `residuals.tsv` / `.refine.json`: the RT and 1/K0 residual sd **before** the
  overwrite is the measurement the refinement was worth (afterwards they are
  zero by construction).
- `.tune.json` → `evaluation.stock.test` vs `evaluation.tuned.test`: the
  held-out (protein-level) error before and after, in deployed units;
  `calibrated_sd` is what a library consumer sees after its own affine refit.

What both outputs are: **per-run objects**. They are right for the run they
were made from and for runs acquired the same way. See
[fine-tuning.md](fine-tuning.md) and [results.md](results.md).
