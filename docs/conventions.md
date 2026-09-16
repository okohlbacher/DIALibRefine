# Conventions

Shared with DIALibGen where the two tools overlap; stated here so that a
reader of one output can read the other.

## Provenance

Every output carries a JSON record of how it was made, twice: as a sidecar
file next to the output and, for Parquet outputs, embedded in the schema
metadata under the key `odia.config_json` — the same key DIALibGen uses, so
a library's history is one chain of records.

| tool | sidecar | contents |
|---|---|---|
| `DIALibraryRefiner` | `<out>.refine.json` | tool + config as applied, SHA-256 of the library and the reference, the reference run's name, every rejection count, gates bypassed, pre-overwrite residuals per axis (mean, sd, p95), units, the per-run warning |
| `DIALibTune` | `<out>.tune.json` | tool, libtorch version, device, recipe, stopping rule, filter and rejection counts, cohorts and their rule, SHA-256 of the stock and tuned models, the course of training, stock and tuned metrics on VAL and TEST, `exported` |

The tuned ONNX itself carries **no** embedded record: it is the stock file
with 21 initializers' bytes replaced, byte-for-byte the same layout, so that
every consumer of the stock model reads it unchanged. The sidecar is the
record; keep it beside the model.

`schema_version` is 1 in every record and is checked on read where a record
is read back (`-config`).

## Configuration JSON (`DIALibraryRefiner`)

`-write_config <file>` writes the effective configuration — every option at
the value the run would use — and `-config <file>` reads one back. Unknown
keys are refused (a typo would otherwise become a default), enumerations
and ranges are validated, and the tool accepts its own `-write_config`
output unchanged. Command-line options outrank the file.

## Modification naming

DIA-NN writes `C(UniMod:4)`; OpenMS writes `C(Carbamidomethyl)`. They are the
same modification, and a verbatim join matches neither to the other —
silently. `DIALibraryRefiner` canonicalises modified sequences to their
UniMod accessions before joining, with balanced-bracket parsing so that
nested forms such as `K(Label:13C(6)15N(2))` resolve (to `UniMod:259`), and
counts the tokens it cannot resolve. On S08 the verbatim join dropped exactly
the cysteine-containing precursors, 9.26 % of the reference.

`DIALibTune` needs no canonicalisation: OpenMS's `AASequence::fromString`
reads the `UniMod:n` form directly, and the encoder is DIALibGen's.

## Units

| quantity | unit | note |
|---|---|---|
| RT in a refined library | minutes of the *reference run* | not iRT; the column's meaning has changed and the tool says so on every run |
| 1/K0 | Vs/cm² | as measured on the reference instrument |
| CCS | Å² | what the CCS model predicts; converted through Mason–Schamp with N₂ (28.0134), 305 K and the 18509 constant, at the precursor's m/z and charge |
| `rt_norm` | RT / `rt_max_minutes` | what the RT model predicts; the scale is in the provenance, not the model |

## Cohorts and hashes

Protein-level hold-outs use `zlib.crc32` of the `Protein.Group` string:
TEST = `crc32(pg) % 5 == 0`; VAL = `crc32("val:" + pg) % 7 == 0` of the rest.
The C++ and Python tools implement the same table-driven CRC-32 and produce
the same cohorts on the same report.

## Per-run objects

A refined library and a tuned model are correct for the run they were made
from and for runs acquired the same way. Nothing prevents copying them
elsewhere; the provenance names the run so that a reader can tell.
