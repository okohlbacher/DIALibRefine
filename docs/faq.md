# FAQ

**Which of the two tools do I want?**
`DIALibraryRefiner` corrects the precursors a run identified and removes the
rest — the paper's reconstruction, no model involved. `DIALibTune` re-trains
the RT and CCS models on those identifications so that *every* predicted
precursor, including the ones the run never saw, gets the run's RT scale and
mobility calibration. For a library-based search of the same run, the tuned
models gave DIA-NN +8 % at matched entrapment budget on S08; the refined
library is the smaller, faster, higher-specificity object. They compose:
tune, regenerate, then refine.

**Why is the tuned model "per run"? Can I reuse it?**
The RT model learns the run's gradient and the CCS model the instrument's
mobility calibration. On another run of the same method they are likely
close; on a different gradient the RT model is wrong by construction.
Cross-run transfer has not been measured here — treat the +8 % as a
same-run number until it has.

**Why does `DIALibTune` refuse to write a model?**
Because no checkpoint beat the stock model on the validation cohort. The
provenance sidecar still records the run. Usual causes: too few
identifications (the cohorts also need ≥ 100 units each), a mislabelled head
(`-head rt` on a CCS file loads — the shapes match — but learns nothing
useful), or an `rt_max_minutes` that does not match the gradient.

**How many identifications do I need?**
Meaningful RT tuning starts around 500 units (0.99 → 0.46 min held-out on
S08) and keeps improving to the full pool (0.32 min). CCS gains are small
and grow with size too. Use everything the run identified at q ≤ 0.01; the
full pool is the default.

**GPU or CPU?**
Cost is optimizer steps, and this model is small: a full-pool, full-horizon
RT run is ~3 700 steps. On a CPU that is ~16 minutes at 4 threads (C++) and
the default stopping rule ends it around epoch 60–75. The GPU numbers are
in [results.md](results.md#gpu). More CPU threads do not help — 4 was the
fastest setting measured, 24–48 were several times slower.

**Does `DIALibTune` need Python?**
No. It is C++ with libtorch; the Python tools under `tools/` are the
reference implementation kept for reproducing the measurements.

**Why is m/z never corrected?**
A precursor's m/z follows from its sequence, charge and modifications; what
drifts is the instrument's mass scale, which belongs to the run and has no
static representation in a library. The paper does not touch it either.

**Why does `-write_intensity` refuse?**
The paper measures intensity replacement as a wash in three separate
figures. Refusing loudly beats running as a no-op.

**The residuals in `.refine.json` are large — is that bad?**
They are measured *before* the overwrite and are the reason the refinement
is worth doing: afterwards they are zero by construction. Large pre-overwrite
residuals mean the predicted library was far from the run.

**`--help` prints `Version: 0.1.0 (OpenMS 3.x)` — which is it?**
The tool's version, then the OpenMS it was built against. Report both.
