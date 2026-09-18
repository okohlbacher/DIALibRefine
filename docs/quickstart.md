# Quickstart

Two tools, one run. Both take a DIA-NN `report.parquet` of the run as the
reference.

## 1. Refine a library against the run

```bash
DIALibRefine -in predicted_library.parquet -ids report.parquet \
                  -out refined_library.parquet -out_report residuals.tsv -write_im
```

Keeps only the precursors the run identified (q ≤ 0.01 at precursor, peptide
and protein level), writes the observed RT — and with `-write_im` the observed
1/K0 for charge ≥ 2 — through, and reports the pre-overwrite residuals.
`refined_library.parquet.refine.json` records everything that happened.

If the reference is a DIA-NN empirical library (`--gen-spec-lib` output),
add `-empirical_library`: it carries no `Global.Q.Value`, so that gate is
bypassed and the bypass recorded.

## 2. Fine-tune the models on the run as well

```bash
DIALibRefine -in predicted_library.parquet -ids report.parquet \
             -out refined_library.parquet -write_im \
             -tune -tune_models models/ -tune_out_models tuned/ \
             -filter:rt_max_minutes 30
```

Same reconstruction as step 1, and then: both heads are trained on `-ids`, and
the **whole** library — including the precursors the run never identified — is
re-predicted through the tuned models. The library comes out of the same `-out`;
there is no second pass through DIALibGen and no `DIALIBGEN_MODEL_DIR` to set.

It prints the stock and tuned held-out error in minutes / 1/K0 per head and
refuses to use a model that did not beat the stock one on validation — in which
case that head's predictions stay as they were. `-tune_out_models` keeps the
tuned ONNX (written in place of the stock one's bytes) and its `.tune.json`
beside it; without it they are discarded after the re-prediction.

`-tune_heads rt` or `ccs` tunes one head only. On a GPU add
`-machine:device cuda:0` for training and `-tune_predict_gpu` for the
re-prediction pass; on a CPU leave `-machine:threads` at 4.

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
