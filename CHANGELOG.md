# Changelog

## [0.1.0] — unreleased

First release. Implements peptide-centric library reconstruction (filter +
observed RT + observed 1/K0) against a reference run's identifications.

### Added
- `DIALibraryRefiner` TOPP tool and the `odia_refine` library.
- Canonical modification naming, so `C(UniMod:4)` and `C(Carbamidomethyl)` join.
  Measured worth on a carbamidomethylated timsTOF run: 3,444 precursors, which
  is exactly the cysteine-containing population.
- Per-axis residual reporting, measured BEFORE the overwrite.
- Refusal of `-write_intensity`, which the source paper measures as a wash.

### Known limitations
- Fragment intensity replacement is not implemented.
- The output is a per-run object; nothing prevents it being copied elsewhere.
- No provenance stamp is written into the Parquet metadata yet, so a refined
  library does not yet declare which run it was refined against.
