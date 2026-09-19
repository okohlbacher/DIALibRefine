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
/// library. The paper does not correct it either.
///
/// **Fragment intensities are replaced only on request, and never blindly.**
/// Until 0.3.0 `-write_intensity` was refused, on the paper's measurement that
/// intensity replacement is a wash. On K562 diaPASEF (DIA-NN 2.0, three
/// replicates) the matched pair said otherwise: DIA-NN refining its OWN library
/// against a run -- which does replace intensities -- reached 8,586 protein
/// groups where this tool's reconstruction, identical in every other respect we
/// could name, reached 7,703. So it is implemented, behind an m/z cross-check
/// that turns a silent mismatch into an error.
///
/// **Nothing fails open.** Revised after the 2026-09-16 review: a gate whose
/// column is missing is an ERROR unless the caller declares the input a
/// pre-filtered empirical library, an unparseable q-value rejects its row rather
/// than passing it, and a join that matches nothing is a failure, not an empty
/// library that looks right.
#pragma once

#include <odia/Library.h>

#include <cmath>
#include <cstdint>
#include <limits>
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
  /// cysteine-containing precursor -- measured: 3,444 of 37,193 on S08, exactly
  /// the cysteine population -- and a verbatim join drops all of them silently.
  ///
  /// Brackets are parsed BALANCED, so `K(Label:13C(6)15N(2))` is one token.
  /// A token that is neither a known name nor a `UniMod:` accession is passed
  /// through verbatim and, when @p unknown is given, counted there.
  std::string canonicalModifiedSequence(std::string_view seq, std::size_t* unknown = nullptr);

  /// One fragment as the reference REPORT describes it, parsed from the
  /// positionally corresponding entries of Fragment.Info, Fragment.Quant.Raw
  /// and Fragment.Correlations.
  struct ObservedFragment
  {
    FragmentType type = FragmentType::Unknown;
    std::uint8_t ordinal = 0;
    std::int8_t charge = 0;       ///< always positive; anything else is an unparseable token
    double mz = 0.0;              ///< the report's own fragment m/z, used ONLY as a cross-check
    float quant = 0.0f;           ///< Fragment.Quant.Raw: an integrated area from ONE run
    float correlation = std::numeric_limits<float>::quiet_NaN();   ///< may be 0 or negative; NaN = column absent
  };

  /// Parse one report row's fragment triple, e.g.
  ///   info  "b9^1/668.3726196;y7^1/659.3471069;"
  ///   quant "367.0187988;79.00454712;"
  ///   corr  "0.7308319807;0.559844017;"
  ///
  /// Returns false, writing nothing, when the lists disagree in length: the
  /// columns are then not positionally paired and every value parsed from them
  /// would land on the wrong fragment. An unparseable TOKEN is skipped and
  /// counted in @p bad_tokens; a length disagreement is a ROW failure. An empty
  /// @p corr means the column is absent and every correlation is NaN.
  bool parseFragmentInfo(std::string_view info, std::string_view quant, std::string_view corr,
                         std::vector<ObservedFragment>& out, std::size_t* bad_tokens = nullptr);

  /// One reference observation of one precursor.
  struct Observation
  {
    float rt = std::numeric_limits<float>::quiet_NaN();   ///< observed, in the reference run's own units
    float im = std::numeric_limits<float>::quiet_NaN();   ///< observed 1/K0; NaN when the reference has none
    float q = 1.0f;                                       ///< precursor q-value
    float pep = 1.0f;                                     ///< posterior error probability, when present
    float evidence = 0.0f;                                ///< search discriminant, when present
    std::uint32_t fragments = 0;                          ///< DISTINCT fragment identities among passing rows; 0 = unknown
    /// Empty unless intensities are being written. Lives HERE, not in a side
    /// map, so that deduplication replaces RT, 1/K0 and the fragment vector
    /// together: a precursor's intensities and its retention time must come from
    /// the same elution event, and splitting them is the one way this feature
    /// can be silently, plausibly wrong.
    std::vector<ObservedFragment> frags;
  };

  struct RefineParams
  {
    /// Delete precursors the reference did not identify.
    ///
    /// This is the paper's largest single lever, not a tidying step: library
    /// specificity accounts for 87% of reproducibility variance in OpenSWATH
    /// and 64% in DIA-NN, and filtering alone beats their transfer-learning arm.
    bool filter = true;

    double q_precursor = 0.01;
    double q_global = 0.01;    ///< peptide-level, across runs
    double q_protein = 0.01;

    /// The reference is a DIA-NN REPORT and must carry every enabled gate's
    /// column and a Decoy column. When false the input is declared a
    /// pre-filtered empirical library: missing gates are BYPASSED, and each
    /// bypass is recorded in the stats and the output's provenance.
    bool require_gates = true;

    /// Minimum DISTINCT reference fragments for a precursor to survive.
    ///
    /// 0 = off, and off is the default. Non-zero requires the reference to
    /// carry fragment identities (Fragment.Type / Series.Number / Charge);
    /// counting report ROWS instead would let six duplicate observations pass
    /// as six fragments. pyprophet couples this gate to intensity writing; here
    /// it is independent so filter-only and reconstructed arms compare the same
    /// precursor set.
    std::size_t min_fragments = 0;

    bool write_rt = true;
    bool write_im = false;          ///< opt-in; see the warning in refine()
    /// Replace predicted fragment intensities with the reference run's observed
    /// ones. Needs Fragment.Info and Fragment.Quant.Raw in -ids, which DIA-NN
    /// writes only under --report-lib-info.
    bool write_intensity = false;

    /// A fragment is TRUSTED when quant > 0 and correlation > this. The default
    /// asks only that the fragment's extracted profile co-elutes with the
    /// precursor at all: an anti-correlated one is interference, and writing its
    /// area in teaches the next search to expect that interference. <= -1
    /// disables the test (quant > 0 only) -- the arm that separates "different
    /// values" from "fewer transitions".
    double intensity_min_correlation = 0.0;

    /// true:  a replaced precursor is RESTRICTED to its trusted transitions.
    /// false: a precursor is replaced only when EVERY one of its transitions is
    ///        trusted, otherwise it keeps its predictions whole. Transition
    ///        counts are then identical to the input, so a benchmark delta is
    ///        attributable to the VALUES alone.
    bool intensity_restrict = true;

    /// Re-sort a replaced precursor's transitions by descending intensity.
    bool intensity_rerank = true;

    /// A replaced precursor keeps at least this many transitions or keeps its
    /// predictions whole. 3 is the bar DIALibGen's digest and decoy stages hold.
    std::size_t intensity_min_fragments = 3;

    /// How an observed area becomes a library intensity.
    ///
    /// LibraryMax is the default because base peak = 1 is NOT an invariant of
    /// these libraries: measured on 300,000 precursors of a DIALibGen 0.10.1
    /// library, 89.7% carry a maximum of 1.0 and the rest as little as 0.04,
    /// since the fragment m/z window and the top-N cut can both exclude the true
    /// base peak. Scaling the observed vector so its maximum equals the maximum
    /// that precursor ALREADY held preserves whatever convention each one has,
    /// and is the only choice that cannot introduce a second scale.
    enum class IntensityNorm { LibraryMax, BasePeak, Sum, Raw };
    IntensityNorm intensity_norm = IntensityNorm::LibraryMax;

    /// Drop a normalised value below this fraction of the kept set's maximum.
    /// Weaker than the generator's floor on purpose: that one is relative to the
    /// full model spectrum's peak, which does not exist here.
    double intensity_min_relative = 1e-4;

    /// A match counts only when the report's fragment m/z agrees with the
    /// library's. THIS is what catches a mis-parse, a different ordinal
    /// convention, and an -in that is not the library the run was searched
    /// against -- every other counter looks healthy in all three cases.
    double intensity_mz_tol_ppm = 20.0;

    /// Refuse when more than this fraction of candidate matches fail that check.
    /// 1.0 surveys a suspect pairing without failing.
    double intensity_max_mz_mismatch = 0.01;

    /// -write_intensity with the filter off leaves two intensity provenances in
    /// one file. Refused unless this is set, as rt_unit=minmax is.
    bool allow_mixed_intensity = false;

    enum class RtUnit
    {
      Observed,   ///< the reference's own units, written through unchanged
      MinMax      ///< rescaled to [0,100] over the matched set; refused with filter off (mixed scales)
    };
    RtUnit rt_unit = RtUnit::Observed;

    /// Which observation wins when a precursor was seen more than once.
    ///
    /// LowestQ ranks on the search's own error estimate (ties: lower PEP, then
    /// first seen). HighestEvidence ranks on DIA-NN's `Evidence` discriminant
    /// and errors if the column is absent. Neither ranks on abundance: the
    /// previous default did, and a brighter wrong-RT observation would win.
    enum class Dedup { LowestQ, HighestEvidence };
    Dedup dedup = Dedup::LowestQ;

    /// Charge states below this never receive an observed 1/K0.
    ///
    /// Charge 1 is censored on a timsTOF diaPASEF method: 78% of S08's z1
    /// precursors sit within 0.02 of the ramp top and their observed value is
    /// the edge of the scan window, not the ion. Writing it into a library
    /// propagates an instrument setting as if it were data.
    int im_min_charge = 2;

    /// The instrument's mobility ramp top (1/K0), if known. Observations within
    /// @ref im_ramp_margin of it are treated as censored and not written.
    /// 0 = unknown. This is an INSTRUMENT limit supplied by the caller, never
    /// the maximum of the data -- a subsample's maximum is not a boundary.
    double im_ramp_top = 0.0;
    double im_ramp_margin = 0.02;

    /// Refuse when fewer than this fraction of passing reference precursors
    /// match the library. 0 = refuse only when NOTHING matches. A targeted
    /// library legitimately covers a small fraction; set this per use.
    double min_match_fraction = 0.0;
  };

  struct RefineStats
  {
    std::string run;                  ///< the reference run, when the report names one
    std::size_t ids_rows = 0;         ///< rows read from the reference
    std::size_t ids_precursors = 0;   ///< distinct precursors in it
    std::size_t ids_passing = 0;      ///< after the gates
    std::size_t ids_decoy = 0;
    std::size_t ids_q_invalid = 0;    ///< rows whose q-values were missing or non-numeric: REJECTED
    std::size_t ids_q_above = 0;
    std::size_t ids_charge_invalid = 0;
    std::size_t ids_unmatched = 0;    ///< observed but absent from the library
    std::size_t ids_ramp_censored = 0;
    std::size_t ids_unknown_mod_tokens = 0;
    std::vector<std::string> gates_bypassed;   ///< names of gates whose column was absent (empirical mode only)

    std::size_t library_before = 0;
    std::size_t library_after = 0;
    std::size_t matched = 0;          ///< TARGET precursors that matched
    std::size_t decoys_kept = 0;      ///< decoys kept because their target was
    std::size_t rt_written = 0;
    std::size_t rt_missing = 0;       ///< matched, write_rt on, but the reference had no finite RT: left as predicted
    std::size_t im_written = 0;
    std::size_t im_missing = 0;
    std::size_t im_charge_excluded = 0;

    /// Residual of the library's PREDICTION against the observation over the
    /// matched TARGETS, before anything was overwritten. NaN when undefined
    /// (n < 2). p95 is the lower nearest-rank quantile of |residual|.
    double rt_resid_mean = std::numeric_limits<double>::quiet_NaN();
    double rt_resid_sd = std::numeric_limits<double>::quiet_NaN();
    double rt_resid_p95 = std::numeric_limits<double>::quiet_NaN();
    double im_resid_mean = std::numeric_limits<double>::quiet_NaN();
    double im_resid_sd = std::numeric_limits<double>::quiet_NaN();
    double im_resid_p95 = std::numeric_limits<double>::quiet_NaN();
    std::size_t rt_resid_n = 0, im_resid_n = 0;

    /// matched / ids_passing. Reported always; enforced against min_match_fraction.
    double match_fraction = 0.0;

    // Intensity replacement. Every rejection has its own counter, so the gates'
    // effect is REPORTED rather than inferred from a total.
    std::size_t intensity_candidate_precursors = 0;   ///< matched targets whose observation carried fragments
    std::size_t intensity_candidate_transitions = 0;  ///< library transitions of those candidates: what the four fates below must sum to
    std::size_t intensity_replaced_precursors = 0;    ///< targets that received observed values
    std::size_t intensity_replaced_decoys = 0;
    std::size_t intensity_kept_predicted = 0;         ///< candidates that kept their predictions whole
    std::size_t intensity_decoy_asymmetry = 0;        ///< reverted because a decoy would have fallen below the bar
    std::size_t intensity_duplicate_key = 0;          ///< more than one TARGET on a key: skipped, not guessed
    std::size_t intensity_matched_transitions = 0;    ///< identity AND m/z agreed
    std::size_t intensity_mz_mismatch = 0;            ///< identity agreed, m/z did not
    std::size_t intensity_unmatched_in_library = 0;   ///< library transition the report does not list
    std::size_t intensity_loss_bearing = 0;           ///< neutral-loss transition: the report cannot describe it
    std::size_t intensity_observed_not_in_library = 0;///< report fragment the library never carried: NEVER added
    std::size_t intensity_gated_zero_quant = 0;
    std::size_t intensity_gated_correlation = 0;
    std::size_t intensity_gated_floor = 0;
    std::size_t intensity_bad_tokens = 0;
    std::size_t intensity_row_length_mismatch = 0;    ///< the three fragment columns disagreed in length
    double intensity_mz_mismatch_fraction = 0.0;
    /// Fraction of replaced precursors whose observed base peak was already the
    /// library's top-ranked transition. Near 1 means replacement is close to a
    /// no-op on this data -- the measurement the old refusal cited a paper for.
    double intensity_rank_agreement = std::numeric_limits<double>::quiet_NaN();
    double intensity_transitions_before = std::numeric_limits<double>::quiet_NaN();   ///< mean, replaced precursors
    double intensity_transitions_after = std::numeric_limits<double>::quiet_NaN();
  };

  class LibraryRefiner
  {
  public:
    using ObsMap = std::unordered_map<std::string, Observation>;

    /// `modified_sequence/charge`, canonicalised.
    static std::string key(std::string_view modified_sequence, int charge);

    /// Read reference observations from a DIA-NN report or a DIA-NN-dialect
    /// library (Parquet). A library-shaped input has one row per transition,
    /// so rows are collapsed to precursors here. Throws on any violation of the
    /// input contract described in RefineParams.
    static ObsMap readObservations(const std::string& path,
                                   const RefineParams& p,
                                   RefineStats& stats);

    /// Apply the observations to @p library in place. Throws when the join
    /// matches nothing (or too little), or when a requested transform cannot
    /// be applied consistently.
    static void refine(Library& library, const ObsMap& obs,
                       const RefineParams& p, RefineStats& stats);
  };
}
