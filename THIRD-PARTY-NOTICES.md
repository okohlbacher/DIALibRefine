# Third-party notices

DIALibRefine is BSD-3-Clause (see `LICENSE`). It vendors no third-party code and
no model weights.

## Method provenance

The refinement implemented here follows the peptide-centric library
reconstruction described in:

> Charkow J., Ghaznavi M., Seale B., Peng J., Gingras A.-C., Röst H.L.
> *Reference-Based Library Construction Improves Performance in low-input
> diaPASEF Workflows.* bioRxiv 10.64898/2026.04.29.721088 (2026).

The authors' own implementation is at
https://github.com/jcharkow/reference-based-libraries and delegates to
`pyprophet library` (PyProphet ≥ 3.0.2). This is an independent reimplementation
against the published description, not a port; where it deviates deliberately —
decoupling the fragment-count gate from intensity writing, and selecting the
de-duplication winner by the ranking quantity rather than by ascending
(q, intensity) — the header comments say so and why.

## Build dependencies

Linked against, not vendored:

| Component | Licence |
|---|---|
| DIALibGen | BSD-3-Clause |
| OpenMS | BSD-3-Clause |
| Apache Arrow / Parquet | Apache-2.0 |
| nlohmann/json | MIT |

DIALibGen brings ONNX Runtime (MIT) and, transitively through OpenMS, further
dependencies with their own terms including Coin-OR components under EPL-2.0.
Consult your OpenMS distribution.
