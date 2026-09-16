# Contributing

## Building for development

The build recipe is the one CI uses, so a green local build is a green CI
build: the conda environment and cmake invocation in
[docs/install.md](docs/install.md#from-source), with `-DDLR_BUILD_FINETUNE=ON`
and `-DDLR_PEPTDEEP_ONNX_DIR` set so that every test is registered.

Against a local DIALibGen checkout:
`-DFETCHCONTENT_SOURCE_DIR_DIALIBGEN=/path/to/DIALibGen` (must be at the
pinned tag or a compatible commit).

## Test tiers

`ctest --test-dir build --output-on-failure` runs all of them; a test whose
prerequisites are missing is *skipped* (exit 77), never silently absent — CI
asserts the three data-dependent ones were registered.

| tier | what | needs |
|---|---|---|
| unit | `canonical_modseq`: the join key, nested modifications, unknown tokens | nothing |
| tool | `--help`, `-write_ini`, `-write_config` round trip, refusals (unknown config key, `-write_intensity`) | the built tools |
| parity | `tune_parity`: the libtorch transcription vs ONNX Runtime on real peptides (RT ≤ 1e-5, CCS ≤ 5e-3), byte-identical write-back, a perturbed model surviving store→load exactly | the stock models |
| standalone | `standalone_test.sh`: both tools in `env -i`, own version, update check off, `-write_ctd` | the built tools |
| e2e | `tune_e2e.sh`: a synthetic single-run report (`synth_report.py`), both heads fine-tuned for 20 epochs, must beat stock on validation, output byte-size-identical to the stock ONNX | the stock models, python + pyarrow + numpy |

Anything that changes the trainer must keep `tune_e2e` green and should be
checked against the Python reference on a real report: cohort counts and
stock metrics must match to the printed digit (they do on S08; see
`docs/results.md`), and a 100-epoch run must land within the replicate sd
(0.004 min RT) of `tools/finetune.py`'s.

## Adversarial review

Non-trivial changes to the trainer or the ONNX writer get a read-only review
by a second model before merging (the C++ port was reviewed by GPT-6-Astra at
effort ultra, sources pasted, and sixteen findings were verified and fixed —
`git log` has the list). Reviewers are confidently wrong sometimes: verify
each finding against the code before acting on it.

## Style

Match the surrounding code. Comments say *why* — the measurement or the
failure that motivated a line — not what the line does. Every number in the
docs carries its date, node and artefact path or it is not a result.

## Releasing

[docs/release.md](docs/release.md): bump, dry-run the packaging by dispatch,
tag. Secrets are added through the GitHub web UI only.
