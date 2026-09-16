// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// DIALibraryRefiner: library + reference identifications -> refined library.
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

#include <OpenMS/APPLICATIONS/TOPPBase.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace
{
  json effectiveConfig(const ODIA::RefineParams& p)
  {
    return json{
      {"schema_version", 1},
      {"filter", p.filter},
      {"q_precursor", p.q_precursor},
      {"q_global", p.q_global},
      {"q_protein", p.q_protein},
      {"min_fragments", p.min_fragments},
      {"write_rt", p.write_rt},
      {"write_im", p.write_im},
      {"write_intensity", p.write_intensity},
      {"rt_unit", p.rt_unit == ODIA::RefineParams::RtUnit::Observed ? "observed" : "minmax"},
      {"dedup", p.dedup == ODIA::RefineParams::Dedup::BestQuality ? "best_quality" : "lowest_q"},
      {"im_ramp_guard", p.im_ramp_guard}};
  }
}

class DIALibraryRefiner final : public OpenMS::TOPPBase
{
public:
  DIALibraryRefiner()
    : TOPPBase("DIALibraryRefiner",
               "Refine a DIA library against a reference run's identifications.",
               false) {}

protected:
  void registerOptionsAndFlags_() override
  {
    registerInputFile_("in", "<file>", "", "Library to refine (DIA-NN Parquet or TSV).");
    setValidFormats_("in", {"parquet", "tsv"}, false);

    registerInputFile_("ids", "<file>", "",
                       "Reference identifications: a DIA-NN report.parquet, or a DIA-NN "
                       "empirical library. Modification naming is canonicalised, so "
                       "C(UniMod:4) and C(Carbamidomethyl) join correctly -- a verbatim "
                       "join silently drops every cysteine precursor on an alkylated sample.");
    setValidFormats_("ids", {"parquet"}, false);

    registerOutputFile_("out", "<file>", "",
                        "Refined library. .parquet carries the recipe in its schema "
                        "metadata; .tsv is the DIA-NN dialect and cannot.");
    setValidFormats_("out", {"parquet", "tsv"}, false);

    registerOutputFile_("out_report", "<file>", "",
                        "Per-axis residual report (TSV): what the library's predictions were "
                        "worth against the reference, measured BEFORE they were overwritten.",
                        false);
    setValidFormats_("out_report", {"tsv"}, false);

    registerInputFile_("config", "<file>", "", "JSON configuration.", false);
    setValidFormats_("config", {"json"}, false);
    registerOutputFile_("write_config", "<file>", "",
                        "Write the effective config here and exit.", false);
    setValidFormats_("write_config", {"json"}, false);

    registerFlag_("no_filter", "Keep precursors the reference did not identify. "
                               "Filtering is the paper's largest single lever; off is a "
                               "declared arm, not a default.");
    registerFlag_("write_im", "Also overwrite 1/K0 with the observed value. OFF by default: "
                              "in this project a library-side mobility rewrite was measured at "
                              "-34.7% end to end, mediated by the mass gate rather than the "
                              "mobility gate.");
    registerFlag_("write_intensity", "Overwrite fragment intensities. NOT IMPLEMENTED; refused.");
    registerDoubleOption_("q_precursor", "<q>", 0.01, "Precursor q-value gate.", false);
    registerDoubleOption_("q_global", "<q>", 0.01, "Global/peptide q-value gate.", false);
    registerDoubleOption_("q_protein", "<q>", 0.01, "Protein q-value gate.", false);
    registerIntOption_("min_fragments", "<n>", 0,
                       "Minimum reference fragments per precursor. 0 = off.", false);
    registerStringOption_("rt_unit", "<unit>", "observed",
                          "observed = the reference's own units; minmax = rescaled to 0..100.",
                          false);
    setValidStrings_("rt_unit", {"observed", "minmax"});
    registerStringOption_("dedup", "<rule>", "best_quality",
                          "Which observation wins when a precursor was seen more than once.",
                          false);
    setValidStrings_("dedup", {"best_quality", "lowest_q"});
    registerDoubleOption_("im_ramp_guard", "<1/K0>", 0.0,
                          "Drop precursors within this much of the observed mobility ramp top, "
                          "where the value is censored rather than measured. 0 = off.", false);
  }

  void apply_(const json& j, ODIA::RefineParams& p)
  {
    const json ref = effectiveConfig(p);
    for (const auto& [k, v] : j.items())
    { if (!ref.contains(k)) { throw std::runtime_error("unknown config key: " + k); } }

    if (j.contains("filter")) { p.filter = j["filter"].get<bool>(); }
    if (j.contains("q_precursor")) { p.q_precursor = j["q_precursor"].get<double>(); }
    if (j.contains("q_global")) { p.q_global = j["q_global"].get<double>(); }
    if (j.contains("q_protein")) { p.q_protein = j["q_protein"].get<double>(); }
    if (j.contains("min_fragments")) { p.min_fragments = j["min_fragments"].get<std::size_t>(); }
    if (j.contains("write_rt")) { p.write_rt = j["write_rt"].get<bool>(); }
    if (j.contains("write_im")) { p.write_im = j["write_im"].get<bool>(); }
    if (j.contains("write_intensity")) { p.write_intensity = j["write_intensity"].get<bool>(); }
    if (j.contains("im_ramp_guard")) { p.im_ramp_guard = j["im_ramp_guard"].get<double>(); }
    if (j.contains("rt_unit"))
    {
      p.rt_unit = j["rt_unit"].get<std::string>() == "minmax"
                    ? ODIA::RefineParams::RtUnit::MinMax
                    : ODIA::RefineParams::RtUnit::Observed;
    }
    if (j.contains("dedup"))
    {
      p.dedup = j["dedup"].get<std::string>() == "lowest_q"
                  ? ODIA::RefineParams::Dedup::LowestQ
                  : ODIA::RefineParams::Dedup::BestQuality;
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
    p.write_im = getFlag_("write_im");
    p.write_intensity = getFlag_("write_intensity");
    p.im_ramp_guard = getDoubleOption_("im_ramp_guard");
    p.rt_unit = getStringOption_("rt_unit") == "minmax"
                  ? ODIA::RefineParams::RtUnit::MinMax
                  : ODIA::RefineParams::RtUnit::Observed;
    p.dedup = getStringOption_("dedup") == "lowest_q"
                ? ODIA::RefineParams::Dedup::LowestQ
                : ODIA::RefineParams::Dedup::BestQuality;

    if (const std::string cfg = getStringOption_("config"); !cfg.empty())
    {
      std::ifstream in(cfg);
      if (!in) { writeLogError_("cannot read config: " + cfg); return INPUT_FILE_NOT_FOUND; }
      try { apply_(json::parse(in, nullptr, true, true), p); }
      catch (const std::exception& e)
      { writeLogError_(std::string("config: ") + e.what()); return ILLEGAL_PARAMETERS; }
    }

    const json eff = effectiveConfig(p);
    if (const std::string wc = getStringOption_("write_config"); !wc.empty())
    {
      std::ofstream(wc) << eff.dump(2) << '\n';
      writeLogInfo_("wrote effective config to " + wc);
      return EXECUTION_OK;
    }

    const std::string in = getStringOption_("in");
    const std::string ids = getStringOption_("ids");
    const std::string out = getStringOption_("out");
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
      writeLogInfo_("reference: " + std::to_string(st.ids_rows) + " rows, " +
                    std::to_string(st.ids_precursors) + " precursors, " +
                    std::to_string(st.ids_passing) + " passing the q-value gates");
      if (st.ids_ramp_dropped)
      { writeLogInfo_("mobility ramp guard dropped " + std::to_string(st.ids_ramp_dropped)); }

      ODIA::LibraryRefiner::refine(library, obs, p, st);
    }
    catch (const std::exception& e)
    { writeLogError_(std::string("refine: ") + e.what()); return UNEXPECTED_RESULT; }

    if (st.join_looks_broken)
    {
      writeLogError_("matched " + std::to_string(st.matched) + " of " +
                     std::to_string(st.ids_passing) + " reference precursors -- under 5%. "
                     "That is a join failure, not a disagreement between library and run. "
                     "Check that the two sides used the same alkylation state: a library "
                     "built without carbamidomethyl cannot match an alkylated sample's "
                     "cysteine precursors at all.");
      return UNEXPECTED_RESULT;
    }

    std::ostringstream m;
    m.setf(std::ios::fixed);
    m << "matched " << st.matched << " of " << st.ids_passing << " reference precursors ("
      << std::setprecision(1) << (st.ids_passing ? 100.0 * static_cast<double>(st.matched)
                                                     / static_cast<double>(st.ids_passing) : 0.0)
      << "%); " << st.ids_unmatched << " observed but absent from the library";
    writeLogInfo_(m.str());

    if (st.rt_resid_n)
    {
      std::ostringstream r;
      r.setf(std::ios::fixed); r.precision(4);
      r << "RT residual BEFORE refinement (library prediction - observed), n=" << st.rt_resid_n
        << ": mean " << st.rt_resid_mean << ", sd " << st.rt_resid_sd
        << ", p95 |resid| " << st.rt_resid_p95;
      writeLogInfo_(r.str());
    }
    if (st.im_resid_n)
    {
      std::ostringstream r;
      r.setf(std::ios::fixed); r.precision(5);
      r << "1/K0 residual BEFORE refinement, n=" << st.im_resid_n
        << ": mean " << st.im_resid_mean << ", sd " << st.im_resid_sd
        << ", p95 |resid| " << st.im_resid_p95;
      writeLogInfo_(r.str());
    }
    writeLogInfo_("wrote " + std::to_string(st.rt_written) + " RT and " +
                  std::to_string(st.im_written) + " 1/K0 values");
    writeLogInfo_("library " + std::to_string(st.library_before) + " -> " +
                  std::to_string(st.library_after) + " precursors (" +
                  std::to_string(st.matched) + " targets + " +
                  std::to_string(st.decoys_kept) + " decoys)" +
                  (p.filter ? "" : " (filter off)"));

    if (p.write_rt)
    {
      writeLogInfo_("NOTE: the RT column now holds the REFERENCE RUN's observed retention "
                    "times, not iRT. This library is a per-run object: it is correct for that "
                    "run and for runs on the same gradient, and wrong elsewhere. Transferring "
                    "a per-run RT model across runs has cost 2,027 precursors here before.");
    }

    try
    {
      ODIA::DIANNLibraryFile::store(out, library);
      writeLogInfo_("wrote " + out);
    }
    catch (const std::exception& e)
    { writeLogError_(std::string("write: ") + e.what()); return CANNOT_WRITE_OUTPUT_FILE; }

    if (const std::string rep = getStringOption_("out_report"); !rep.empty())
    {
      std::ofstream o(rep);
      o << "metric\tvalue\n";
      o << "ids_rows\t" << st.ids_rows << '\n'
        << "ids_precursors\t" << st.ids_precursors << '\n'
        << "ids_passing\t" << st.ids_passing << '\n'
        << "ids_unmatched\t" << st.ids_unmatched << '\n'
        << "ids_ramp_dropped\t" << st.ids_ramp_dropped << '\n'
        << "library_before\t" << st.library_before << '\n'
        << "library_after\t" << st.library_after << '\n'
        << "matched\t" << st.matched << '\n'
        << "decoys_kept\t" << st.decoys_kept << '\n'
        << "rt_written\t" << st.rt_written << '\n'
        << "im_written\t" << st.im_written << '\n';
      o.setf(std::ios::fixed);
      o << std::setprecision(6)
        << "rt_resid_n\t" << st.rt_resid_n << '\n'
        << "rt_resid_mean\t" << st.rt_resid_mean << '\n'
        << "rt_resid_sd\t" << st.rt_resid_sd << '\n'
        << "rt_resid_p95\t" << st.rt_resid_p95 << '\n'
        << "im_resid_n\t" << st.im_resid_n << '\n'
        << "im_resid_mean\t" << st.im_resid_mean << '\n'
        << "im_resid_sd\t" << st.im_resid_sd << '\n'
        << "im_resid_p95\t" << st.im_resid_p95 << '\n';
      writeLogInfo_("wrote report to " + rep);
    }

    return EXECUTION_OK;
  }
};

int main(int argc, const char** argv)
{
  DIALibraryRefiner tool;
  return tool.main(argc, argv);
}
