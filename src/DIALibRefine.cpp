// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// DIALibRefine: library + reference identifications -> refined library.
///
/// The counterpart to DIALibGen. That tool predicts a library from a FASTA;
/// this one replaces those predictions with what a reference run measured, and
/// deletes the hypotheses it never saw.
///
/// Like DIALibGen, this file is a config parser and a call sequence. Every
/// algorithm lives in odia_refine. If it grows a second algorithm it is in the
/// wrong file.

#include <odia/DIANNLibraryFile.h>
#include <odia/Library.h>
#include <odia/LibraryRefiner.h>

#include "ToolBoilerplate.h"

#include <OpenMS/APPLICATIONS/TOPPBase.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace
{
  const char* rtUnitName(ODIA::RefineParams::RtUnit u)
  { return u == ODIA::RefineParams::RtUnit::MinMax ? "minmax" : "observed"; }
  const char* dedupName(ODIA::RefineParams::Dedup d)
  { return d == ODIA::RefineParams::Dedup::HighestEvidence ? "highest_evidence" : "lowest_q"; }

  json effectiveConfig(const ODIA::RefineParams& p)
  {
    return json{
      {"schema_version", 1},
      {"filter", p.filter},
      {"q_precursor", p.q_precursor}, {"q_global", p.q_global}, {"q_protein", p.q_protein},
      {"require_gates", p.require_gates},
      {"min_fragments", p.min_fragments},
      {"write_rt", p.write_rt}, {"write_im", p.write_im}, {"write_intensity", p.write_intensity},
      {"rt_unit", rtUnitName(p.rt_unit)},
      {"dedup", dedupName(p.dedup)},
      {"im_min_charge", p.im_min_charge},
      {"im_ramp_top", p.im_ramp_top}, {"im_ramp_margin", p.im_ramp_margin},
      {"min_match_fraction", p.min_match_fraction}};
  }

  double nanOrValue(double v) { return v; }   // JSON: NaN serialises as null below
  json num(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }
}

class DIALibRefine final : public OpenMS::TOPPBase
{
public:
  DIALibRefine()
    : TOPPBase("DIALibRefine",
               "Refine a DIA library against a reference run's identifications.",
               false) {
#ifdef DLR_VERSION
    // Our own version, not the OpenMS this happened to be built against.
    version_ = DLR_VERSION;
    verboseVersion_ = dlr::verboseVersion();
#endif
  }

protected:
  void registerOptionsAndFlags_() override
  {
    registerInputFile_("in", "<file>", "", "Library to refine (DIA-NN Parquet or TSV).");
    setValidFormats_("in", {"parquet", "tsv"}, false);
    registerInputFile_("ids", "<file>", "",
                       "Reference identifications: a DIA-NN report.parquet, or -- with -empirical_library -- "
                       "a DIA-NN empirical library. Modification naming is canonicalised, so C(UniMod:4) "
                       "and C(Carbamidomethyl) join; a verbatim join silently drops every cysteine precursor "
                       "on an alkylated sample.");
    setValidFormats_("ids", {"parquet"}, false);
    registerOutputFile_("out", "<file>", "",
                        "Refined library. .parquet carries the full refinement recipe and statistics in its "
                        "schema metadata (odia.config_json); .tsv cannot. Either way a <out>.refine.json "
                        "sidecar is written.");
    setValidFormats_("out", {"parquet", "tsv"}, false);
    registerOutputFile_("out_report", "<file>", "",
                        "Per-axis residual report (TSV), measured BEFORE the overwrite.", false);
    setValidFormats_("out_report", {"tsv"}, false);
    registerInputFile_("config", "<file>", "", "JSON configuration; unknown keys and values are errors.", false);
    setValidFormats_("config", {"json"}, false);
    registerOutputFile_("write_config", "<file>", "", "Write the effective config here and exit.", false);
    setValidFormats_("write_config", {"json"}, false);

    registerFlag_("no_filter", "Keep precursors the reference did not identify. Filtering is the paper's "
                               "largest single lever; off is a declared arm, not a default.");
    registerFlag_("empirical_library", "Declare -ids a pre-filtered empirical library rather than a report: gates "
                                       "whose columns are absent are BYPASSED and each bypass is recorded. Without "
                                       "this, a missing gate column is an error.");
    registerFlag_("write_im", "Also overwrite 1/K0 with the observed value for charges >= -im_min_charge. OFF by "
                              "default: a library-side mobility rewrite once measured -34.7% end to end in ODIA.");
    registerFlag_("write_intensity", "Overwrite fragment intensities. NOT IMPLEMENTED; refused.");
    registerDoubleOption_("q_precursor", "<q>", 0.01, "Precursor q-value gate (>= 1 disables).", false);
    registerDoubleOption_("q_global", "<q>", 0.01, "Global/peptide q-value gate (>= 1 disables).", false);
    registerDoubleOption_("q_protein", "<q>", 0.01, "Protein q-value gate (>= 1 disables).", false);
    registerIntOption_("min_fragments", "<n>", 0,
                       "Minimum DISTINCT reference fragments per precursor; needs fragment identities in -ids. 0 = off.", false);
    registerStringOption_("rt_unit", "<unit>", "observed",
                          "observed = the reference's own units; minmax = rescaled to 0..100 over the matched set "
                          "(refused with -no_filter).", false);
    setValidStrings_("rt_unit", {"observed", "minmax"});
    registerStringOption_("dedup", "<rule>", "lowest_q",
                          "Which observation wins when a precursor was seen more than once: lowest_q (ties: lower "
                          "PEP, then first seen) or highest_evidence (needs an Evidence column).", false);
    setValidStrings_("dedup", {"lowest_q", "highest_evidence"});
    registerIntOption_("im_min_charge", "<z>", 2, "Charges below this never receive an observed 1/K0 (z1 is censored "
                                                  "at the ramp top on timsTOF diaPASEF).", false);
    registerDoubleOption_("im_ramp_top", "<1/K0>", 0.0, "The instrument's mobility ramp top, if known; observations "
                                                        "within -im_ramp_margin of it are treated as censored. 0 = unknown.", false);
    registerDoubleOption_("im_ramp_margin", "<1/K0>", 0.02, "See -im_ramp_top.", false);
    registerDoubleOption_("min_match_fraction", "<f>", 0.0,
                          "Refuse when fewer than this fraction of passing reference precursors match the library. "
                          "0 = refuse only when nothing matches.", false);
  }

  static void applyJson_(const json& j, ODIA::RefineParams& p)
  {
    const json ref = effectiveConfig(p);
    for (const auto& [k, v] : j.items())
    { if (!ref.contains(k)) { throw std::runtime_error("unknown config key: " + k); } }
    if (j.contains("schema_version") && j["schema_version"] != 1)
    { throw std::runtime_error("unsupported schema_version " + j["schema_version"].dump() + "; this tool writes 1"); }

    auto get_bool = [&](const char* k, bool& dst) { if (j.contains(k)) { dst = j.at(k).get<bool>(); } };
    auto get_num = [&](const char* k, double& dst, double lo, double hi)
    {
      if (!j.contains(k)) { return; }
      const double v = j.at(k).get<double>();
      if (!(v >= lo && v <= hi)) { throw std::runtime_error(std::string(k) + " out of range"); }
      dst = v;
    };
    get_bool("filter", p.filter); get_bool("require_gates", p.require_gates);
    get_bool("write_rt", p.write_rt); get_bool("write_im", p.write_im); get_bool("write_intensity", p.write_intensity);
    get_num("q_precursor", p.q_precursor, 0.0, 1.0); get_num("q_global", p.q_global, 0.0, 1.0); get_num("q_protein", p.q_protein, 0.0, 1.0);
    get_num("im_ramp_top", p.im_ramp_top, 0.0, 10.0); get_num("im_ramp_margin", p.im_ramp_margin, 0.0, 1.0);
    get_num("min_match_fraction", p.min_match_fraction, 0.0, 1.0);
    if (j.contains("min_fragments")) { p.min_fragments = j.at("min_fragments").get<std::size_t>(); }
    if (j.contains("im_min_charge")) { p.im_min_charge = j.at("im_min_charge").get<int>(); if (p.im_min_charge < 1) { throw std::runtime_error("im_min_charge must be >= 1"); } }
    if (j.contains("rt_unit"))
    {
      const auto s = j.at("rt_unit").get<std::string>();
      if (s == "observed") { p.rt_unit = ODIA::RefineParams::RtUnit::Observed; }
      else if (s == "minmax") { p.rt_unit = ODIA::RefineParams::RtUnit::MinMax; }
      else { throw std::runtime_error("rt_unit must be observed or minmax, not '" + s + "'"); }
    }
    if (j.contains("dedup"))
    {
      const auto s = j.at("dedup").get<std::string>();
      if (s == "lowest_q") { p.dedup = ODIA::RefineParams::Dedup::LowestQ; }
      else if (s == "highest_evidence") { p.dedup = ODIA::RefineParams::Dedup::HighestEvidence; }
      else { throw std::runtime_error("dedup must be lowest_q or highest_evidence, not '" + s + "'"); }
    }
  }

  ExitCodes main_(int, const char**) override
  {
    ODIA::RefineParams p;
    p.q_precursor = getDoubleOption_("q_precursor");
    p.q_global = getDoubleOption_("q_global");
    p.q_protein = getDoubleOption_("q_protein");
    p.min_fragments = static_cast<std::size_t>(std::max(0, getIntOption_("min_fragments")));
    p.filter = !getFlag_("no_filter");
    p.require_gates = !getFlag_("empirical_library");
    p.write_im = getFlag_("write_im");
    p.write_intensity = getFlag_("write_intensity");
    p.im_min_charge = getIntOption_("im_min_charge");
    p.im_ramp_top = getDoubleOption_("im_ramp_top");
    p.im_ramp_margin = getDoubleOption_("im_ramp_margin");
    p.min_match_fraction = getDoubleOption_("min_match_fraction");
    p.rt_unit = getStringOption_("rt_unit") == "minmax" ? ODIA::RefineParams::RtUnit::MinMax : ODIA::RefineParams::RtUnit::Observed;
    p.dedup = getStringOption_("dedup") == "highest_evidence" ? ODIA::RefineParams::Dedup::HighestEvidence : ODIA::RefineParams::Dedup::LowestQ;

    if (const std::string cfg = getStringOption_("config"); !cfg.empty())
    {
      std::ifstream in(cfg);
      if (!in) { writeLogError_("cannot read config: " + cfg); return INPUT_FILE_NOT_FOUND; }
      try { applyJson_(json::parse(in, nullptr, true, true), p); }
      catch (const std::exception& e) { writeLogError_(std::string("config: ") + e.what()); return ILLEGAL_PARAMETERS; }
    }

    const json eff = effectiveConfig(p);
    if (const std::string wc = getStringOption_("write_config"); !wc.empty())
    {
      std::ofstream o(wc);
      o << eff.dump(2) << '\n';
      if (!o) { writeLogError_("cannot write " + wc); return CANNOT_WRITE_OUTPUT_FILE; }
      writeLogInfo_("wrote effective config to " + wc);
      return EXECUTION_OK;
    }

    const std::string in = getStringOption_("in"), ids = getStringOption_("ids"), out = getStringOption_("out");
    if (in.empty() || ids.empty() || out.empty())
    { writeLogError_("-in, -ids and -out are required"); return ILLEGAL_PARAMETERS; }

    ODIA::Library library;
    ODIA::RefineStats st;
    try
    {
      ODIA::DIANNLibraryFile::load(in, library);
      writeLogInfo_("library: " + std::to_string(library.precursorCount()) + " precursors, " +
                    std::to_string(library.transitionCount()) + " transitions");
      const auto obs = ODIA::LibraryRefiner::readObservations(ids, p, st);
      writeLogInfo_("reference" + (st.run.empty() ? std::string() : " (run " + st.run + ")") + ": " +
                    std::to_string(st.ids_rows) + " rows, " + std::to_string(st.ids_precursors) + " precursors, " +
                    std::to_string(st.ids_passing) + " passing the gates; rejected: " +
                    std::to_string(st.ids_decoy) + " decoy, " + std::to_string(st.ids_q_invalid) + " invalid q, " +
                    std::to_string(st.ids_q_above) + " above threshold, " + std::to_string(st.ids_charge_invalid) + " bad charge");
      for (const auto& g : st.gates_bypassed)
      { writeLogWarn_("gate " + g + " BYPASSED: the reference has no such column (-empirical_library); recorded in provenance"); }
      if (st.ids_unknown_mod_tokens)
      { writeLogWarn_(std::to_string(st.ids_unknown_mod_tokens) + " modification tokens were neither a known name nor a UniMod accession; passed through verbatim"); }
      if (st.ids_ramp_censored)
      { writeLogInfo_(std::to_string(st.ids_ramp_censored) + " observed 1/K0 values within " + std::to_string(p.im_ramp_margin) +
                      " of the declared ramp top " + std::to_string(p.im_ramp_top) + " treated as censored"); }

      ODIA::LibraryRefiner::refine(library, obs, p, st);
    }
    catch (const std::exception& e) { writeLogError_(std::string("refine: ") + e.what()); return UNEXPECTED_RESULT; }

    std::ostringstream m;
    m.setf(std::ios::fixed); m.precision(1);
    m << "matched " << st.matched << " of " << st.ids_passing << " reference precursors ("
      << 100.0 * st.match_fraction << "%); " << st.ids_unmatched << " observed but absent from the library";
    writeLogInfo_(m.str());
    auto resid = [&](const char* axis, std::size_t n, double mean, double sd, double p95, int prec)
    {
      if (!n) { return; }
      std::ostringstream r; r.setf(std::ios::fixed); r.precision(prec);
      r << axis << " residual BEFORE refinement (library prediction - observed), n=" << n
        << ": mean " << mean << ", sd " << sd << ", p95 |resid| " << p95;
      writeLogInfo_(r.str());
    };
    resid("RT", st.rt_resid_n, st.rt_resid_mean, st.rt_resid_sd, st.rt_resid_p95, 4);
    resid("1/K0", st.im_resid_n, st.im_resid_mean, st.im_resid_sd, st.im_resid_p95, 5);
    writeLogInfo_("wrote " + std::to_string(st.rt_written) + " RT values" +
                  (st.rt_missing ? " (" + std::to_string(st.rt_missing) + " matched targets had no observed RT and keep their prediction)" : "") +
                  " and " + std::to_string(st.im_written) + " 1/K0 values" +
                  (st.im_charge_excluded ? " (" + std::to_string(st.im_charge_excluded) + " below im_min_charge kept their prediction)" : "") +
                  (st.im_missing ? " (" + std::to_string(st.im_missing) + " had no usable observed 1/K0)" : ""));
    writeLogInfo_("library " + std::to_string(st.library_before) + " -> " + std::to_string(st.library_after) +
                  " precursors (" + std::to_string(st.matched) + " targets + " + std::to_string(st.decoys_kept) + " decoys)" +
                  (p.filter ? "" : " (filter off)"));
    if (p.write_rt)
    {
      writeLogInfo_("NOTE: the RT column now holds the REFERENCE RUN's observed retention times (unit: " +
                    std::string(rtUnitName(p.rt_unit)) + "), not iRT. This library is a per-run object.");
    }

    // Provenance: the recipe, the inputs by content hash, the run, the units and every
    // count above -- embedded in the Parquet and always written as a sidecar (M13).
    json prov = {
      {"tool", "DIALibRefine"}, {"tool_version", DLR_VERSION},
      {"config", eff},
      {"inputs", {{"library", std::filesystem::absolute(in).string()}, {"library_sha", ODIA::DIANNLibraryFile::hashFile(in)},
                  {"reference", std::filesystem::absolute(ids).string()}, {"reference_sha", ODIA::DIANNLibraryFile::hashFile(ids)},
                  {"reference_run", st.run}}},
      {"gates_bypassed", st.gates_bypassed},
      {"reference", {{"rows", st.ids_rows}, {"precursors", st.ids_precursors}, {"passing", st.ids_passing},
                     {"decoy", st.ids_decoy}, {"q_invalid", st.ids_q_invalid}, {"q_above", st.ids_q_above},
                     {"charge_invalid", st.ids_charge_invalid}, {"unknown_mod_tokens", st.ids_unknown_mod_tokens},
                     {"ramp_censored", st.ids_ramp_censored}, {"unmatched", st.ids_unmatched}}},
      {"library", {{"before", st.library_before}, {"after", st.library_after}, {"matched_targets", st.matched},
                   {"decoys_kept", st.decoys_kept}, {"match_fraction", st.match_fraction},
                   {"rt_written", st.rt_written}, {"rt_missing", st.rt_missing},
                   {"im_written", st.im_written}, {"im_missing", st.im_missing}, {"im_charge_excluded", st.im_charge_excluded}}},
      {"residual_before", {{"rt", {{"n", st.rt_resid_n}, {"mean", num(st.rt_resid_mean)}, {"sd", num(st.rt_resid_sd)}, {"p95_abs", num(st.rt_resid_p95)}}},
                           {"im", {{"n", st.im_resid_n}, {"mean", num(st.im_resid_mean)}, {"sd", num(st.im_resid_sd)}, {"p95_abs", num(st.im_resid_p95)}}},
                           {"sd_convention", "ddof=1; p95 = lower nearest-rank quantile of |residual|; NaN -> null when n<2"}}},
      {"units", {{"rt", p.write_rt ? (p.rt_unit == ODIA::RefineParams::RtUnit::MinMax ? "0..100 minmax over the matched set" : "the reference run's own RT units") : "unchanged (library prediction)"},
                 {"im", p.write_im ? "observed 1/K0 for charges >= im_min_charge; CCS cleared where written" : "unchanged (library prediction)"}}},
      {"warning", "Per-run object: correct for the reference run and its gradient, wrong elsewhere."}};

    try
    {
      if (out.ends_with(".parquet"))
      { ODIA::DIANNLibraryFile::storeParquetCompact(out, library, ODIA::DIANNLibraryFile::Fingerprint{}, prov.dump()); }
      else if (out.ends_with(".tsv"))
      { ODIA::DIANNLibraryFile::storeTSV(out, library); }
      else { writeLogError_("-out must end in .parquet or .tsv"); return ILLEGAL_PARAMETERS; }
      std::ofstream side(out + ".refine.json");
      side << prov.dump(2) << '\n';
      if (!side) { writeLogError_("cannot write " + out + ".refine.json"); return CANNOT_WRITE_OUTPUT_FILE; }
      writeLogInfo_("wrote " + out + " and " + out + ".refine.json");
    }
    catch (const std::exception& e) { writeLogError_(std::string("write: ") + e.what()); return CANNOT_WRITE_OUTPUT_FILE; }

    if (const std::string rep = getStringOption_("out_report"); !rep.empty())
    {
      std::ofstream o(rep);
      o << "metric\tvalue\n";
      for (const auto& [k, v] : prov["reference"].items()) { o << "ids_" << k << '\t' << v << '\n'; }
      for (const auto& [k, v] : prov["library"].items()) { o << "library_" << k << '\t' << v << '\n'; }
      o.setf(std::ios::fixed); o.precision(6);
      o << "rt_resid_n\t" << st.rt_resid_n << "\nrt_resid_mean\t" << st.rt_resid_mean << "\nrt_resid_sd\t" << st.rt_resid_sd
        << "\nrt_resid_p95\t" << st.rt_resid_p95 << "\nim_resid_n\t" << st.im_resid_n << "\nim_resid_mean\t" << st.im_resid_mean
        << "\nim_resid_sd\t" << st.im_resid_sd << "\nim_resid_p95\t" << st.im_resid_p95 << '\n';
      if (!o) { writeLogError_("cannot write " + rep); return CANNOT_WRITE_OUTPUT_FILE; }
      writeLogInfo_("wrote report to " + rep);
    }
    return EXECUTION_OK;
  }
};

int main(int argc, const char** argv)
{
  dlr::disableUpdateCheckUnlessSet();
  dlr::ToolHandlerRegistration ttd("DIALibRefine");
  DIALibRefine tool;
  return tool.main(argc, argv);
}
