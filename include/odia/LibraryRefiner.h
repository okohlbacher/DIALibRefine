// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// Reference-based library refinement: replace a library's PREDICTED peptide
/// properties with the ones a reference run actually measured, and delete the
/// precursors it never saw.
///
/// The method is Charkow et al., "Reference-Based Library Construction Improves
/// Performance in low-input diaPASEF Workflows" (bioRxiv 10.64898/2026.04.29.721088),
/// there called "peptide-centric library reconstruction". It is deliberately
/// NOT a fit: retention time and 1/K0 are OVERWRITTEN with the observed values,
/// not regressed onto them. There is no model, and therefore no generalisation
/// -- a precursor absent from the reference gets no correction at all, which is
/// why the filter and the replacement always travel together.
///
/// What this refuses to do, and why:
///
/// **m/z is never touched.** A precursor's m/z follows from its sequence,
/// charge and modifications; what drifts is the instrument's mass scale, which
/// is a property of the RUN and has no static-column representation in a
/// library. The paper does not correct it either. If a library's m/z is wrong,
/// it is wrong arithmetic, not miscalibration, and the fix is to recompute it
/// from the sequence rather than to fit anything.
#pragma once

#include <odia/Library.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ODIA
{
  /// Canonical form of a modified sequence, so a DIA-NN report joins an ODIA
  /// library.
  ///
  /// NOT cosmetic. DIA-NN writes `C(UniMod:4)`; a library generated through
  /// OpenMS writes `C(Carbamidomethyl)`. On an alkylated sample that is every
  /// cysteine-containing precursor -- about 10% of the identifications -- and a
  /// verbatim join drops all of them without a word, leaving a result that
  /// looks like a slightly worse search rather than a broken join.
  std::string canonicalModifiedSequence(std::string_view seq);

  /// One reference observation of one precursor.
  struct Observation
  {
    float rt = 0.0f;          ///< observed, in the reference run's own units
    float im = 0.0f;          ///< observed 1/K0; NaN when the reference has none
    float quality = 0.0f;     ///< ranking quantity for de-duplication
    float q = 1.0f;           ///< precursor q-value
    std::uint32_t fragments = 0;  ///< rows seen for this precursor, when known
  };

  struct RefineParams
  {
    /// Delete precursors the reference did not identify.
    ///
    /// This is the paper's largest single lever, not a tidying step: library
    /// specificity accounts for 87% of reproducibility variance in OpenSWATH
    /// and 64% in DIA-NN, and filtering alone beats their transfer-learning arm.
    /// The named mechanism is that OpenSWATH estimates the proportion of nulls
    /// in the library when computing q-values, so a library that is >98% never-
    /// observed hypotheses is being scored against a null it invented.
    bool filter = true;

    double q_precursor = 0.01;
    double q_global = 0.01;    ///< peptide-level, across runs
    double q_protein = 0.01;

    /// Minimum reference fragments for a precursor to survive.
    ///
    /// 0 = off, and off is the default. pyprophet couples this to intensity
    /// writing (`min_fragments` is documented there as "only relevant if
    /// intensity_calibration is True"); the consequence, which the paper's
    /// authors state themselves, is that their filter-only library is LARGER
    /// than their reconstructed one -- so the two arms do not compare the same
    /// precursor set. Decoupled here so they can.
    std::size_t min_fragments = 0;

    bool write_rt = true;
    bool write_im = false;          ///< opt-in; see the warning in refine()
    bool write_intensity = false;   ///< not implemented; refused loudly

    enum class RtUnit
    {
      Observed,   ///< the reference's own units, written through unchanged
      MinMax      ///< rescaled to [0,100], pyprophet's default
    };
    RtUnit rt_unit = RtUnit::Observed;

    /// Which observation wins when a precursor was seen more than once.
    ///
    /// BestQuality ranks on the quantity the search ranked by. pyprophet sorts
    /// ascending on (q, intensity) and keeps the first, which on a q tie keeps
    /// the LOWEST-intensity observation -- the opposite of its own comment.
    /// Selecting by anything other than the ranking quantity has manufactured a
    /// 17% error rate in this project before.
    enum class Dedup { BestQuality, LowestQ };
    Dedup dedup = Dedup::BestQuality;

    /// Drop precursors whose observed 1/K0 sits within this much of the
    /// mobility ramp limit. 0 disables.
    ///
    /// Charge-1 precursors pile up against the top of the ramp, where the
    /// observed value is censored rather than measured; writing that censored
    /// value into a library propagates a boundary artefact as if it were data.
    double im_ramp_guard = 0.0;
  };

  struct RefineStats
  {
    std::size_t ids_rows = 0;         ///< rows read from the reference
    std::size_t ids_precursors = 0;   ///< distinct precursors in it
    std::size_t ids_passing = 0;      ///< after the q-value and fragment gates
    std::size_t ids_unmatched = 0;    ///< observed but absent from the library
    std::size_t ids_ramp_dropped = 0;

    std::size_t library_before = 0;
    std::size_t library_after = 0;
    std::size_t matched = 0;          ///< TARGET precursors that got a value
    std::size_t decoys_kept = 0;      ///< decoys kept because their target was
    std::size_t rt_written = 0;
    std::size_t im_written = 0;

    /// Residual of the library's PREDICTION against the observation, over the
    /// matched precursors, before anything was overwritten. This is the
    /// measurement the refinement is worth -- after the write it is zero by
    /// construction and says nothing.
    double rt_resid_mean = 0.0, rt_resid_sd = 0.0, rt_resid_p95 = 0.0;
    double im_resid_mean = 0.0, im_resid_sd = 0.0, im_resid_p95 = 0.0;
    std::size_t rt_resid_n = 0, im_resid_n = 0;

    /// Set when the join found nothing, which on this data almost always means
    /// the modification naming differed rather than that the run disagreed.
    bool join_looks_broken = false;
  };

  class LibraryRefiner
  {
  public:
    using ObsMap = std::unordered_map<std::string, Observation>;

    /// `modified_sequence/charge`, canonicalised.
    static std::string key(std::string_view modified_sequence, int charge);

    /// Read reference observations from a DIA-NN report or a DIA-NN-dialect
    /// library (both Parquet and TSV). A library-shaped input has one row per
    /// transition, so rows are collapsed to precursors here.
    static ObsMap readObservations(const std::string& path,
                                   const RefineParams& p,
                                   RefineStats& stats);

    /// Apply the observations to @p library in place.
    static void refine(Library& library, const ObsMap& obs,
                       const RefineParams& p, RefineStats& stats);
  };
}
