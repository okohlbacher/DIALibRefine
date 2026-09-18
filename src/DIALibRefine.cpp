// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// DIALibRefine: library + reference identifications -> refined library.
///
/// The counterpart to DIALibGen. That tool predicts a library from a FASTA;
/// this one replaces those predictions with what a reference run measured, and
/// deletes the hypotheses it never saw.
///
/// With -tune it first fine-tunes the RT and CCS models on the same reference
/// and re-predicts the library through them, so the precursors the run did NOT
/// identify are corrected too -- the paper's downstream stage. The output is a
/// library either way; the tuned ONNX files are an intermediate, kept only if
/// -tune:out_models asks for them.
///
/// Like DIALibGen, this file is a config parser and a call sequence. Every
/// algorithm lives in odia_refine. If it grows a second algorithm it is in the
/// wrong file.

#include <odia/DIANNLibraryFile.h>
#include <odia/Library.h>
#include <odia/LibraryGenerator.h>
#include <odia/LibraryRefiner.h>

// The fine-tuning stage is optional at BUILD time: it needs a CXX11-ABI
// libtorch, which linux-arm64 has no usable build of. Trainer.h is torch-free,
// but odia_tune propagates ${TORCH_LIBRARIES} PUBLIC, so the include and the
// calls are guarded rather than the library being linked unconditionally.
#ifdef DLR_WITH_FINETUNE
#include <odia/tune/Trainer.h>
#endif

#include "ToolBoilerplate.h"

#include <OpenMS/APPLICATIONS/TOPPBase.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
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

    // Registered in EVERY build, including one without libtorch, so --help,
    // -write_ini and -write_ctd describe the same tool on every platform. -tune
    // itself is what fails when the stage was not compiled in.
    registerFlag_("tune", "Fine-tune the RT and CCS models on -ids and re-predict the whole library through them "
                          "BEFORE refining, so precursors the reference never identified are corrected too. "
                          "Needs a build with the fine-tuning stage. Turns a seconds-long refinement into a "
                          "training run plus whole-library inference.");
    registerStringOption_("tune_models", "<dir>", "", "Directory holding the stock peptdeep_{rt,ccs}_dynamic.onnx. "
                                                      "Default: $DIALIBGEN_MODEL_DIR.", false);
    registerStringOption_("tune_heads", "<which>", "both", "Which models to tune.", false);
    setValidStrings_("tune_heads", {"rt", "ccs", "both"});
    registerStringOption_("tune_out_models", "<dir>", "", "Keep the tuned ONNX files (and their .tune.json and "
                                                          "and .trajectory.tsv sidecars) here. Default: a scratch "
                                                          "directory, removed on exit -- the deliverable is the library.", false);
    registerFlag_("tune_predict_gpu", "Use the GPU for the re-prediction pass (the ONNX one, not training).");
    registerIntOption_("tune_predict_sessions", "<n>", 0, "ONNX Runtime sessions for the re-prediction pass; 0 = default.", false);
    registerFlag_("tune_keep_free_cysteine_offset",
                  "Keep DIALibGen's free-cysteine RT offset when re-predicting with a TUNED RT model. Off by "
                  "default: that offset was fitted against the STOCK model, and a model tuned on this run's own "
                  "identifications has had the chance to learn the effect itself -- applying both counts it twice.");

    // The recipe, verbatim from DIALibTune so a merged tool costs nobody a knob.
    // Prefixed tune: because -filter:q_value on a tool whose headline job is
    // filtering a library, sitting next to -q_precursor with the same default,
    // is a trap. TOPPBase takes the subsection up to the LAST colon.
    registerTOPPSubsection_("filter", "Fine-tuning: which observations train the model. NOTE: this is the TRAINING-set filter, not the library filter -- that is -q_precursor and friends.");
    registerDoubleOption_("filter:q_value", "<q>", 0.01, "Precursor Q.Value threshold", false);
    registerIntOption_("filter:min_charge", "<z>", 2, "CCS only: lowest precursor charge used (RT collapses all charges). z1 is censored at the mobility ramp top on timsTOF", false);
    setMinInt_("filter:min_charge", 1); setMaxInt_("filter:min_charge", 8);
    registerFlag_("filter:allow_z1", "CCS: permit tune:filter:min_charge 1 (censored observations enter training)");
    registerDoubleOption_("filter:rt_spread_max", "<min>", 0.2, "Drop a sequence whose charge states' RTs span more than this (minutes)", false);
    registerDoubleOption_("filter:rt_max_minutes", "<min>", 0.0, "rt_norm denominator; 0 = the run's maximum observed RT", false);

    registerTOPPSubsection_("cohort", "Fine-tuning: protein-level cohorts (frozen before subsampling)");
    registerIntOption_("cohort:train_size", "<n>", 0, "Training units (sequences for rt, sequence x charge for ccs); 0 = full pool", false);
    registerDoubleOption_("cohort:train_frac", "<f>", 0.0, "Alternative to train_size: fraction of the pool", false);
    registerFlag_("cohort:no_inner_val", "No inner validation cohort: its units rejoin the pool and checkpoints are selected on TEST, which is then no longer a held-out number");

    registerTOPPSubsection_("train", "Fine-tuning: recipe");
    registerIntOption_("train:epochs", "<n>", 100, "Horizon of the cosine schedule (and the maximum epochs)", false);
    registerIntOption_("train:warmup", "<n>", 10, "Linear warmup epochs", false);
    registerDoubleOption_("train:lr", "<lr>", 1e-4, "Peak learning rate (Adam)", false);
    registerIntOption_("train:batch_size", "<n>", 1024, "Batch size within a length group", false);

    registerTOPPSubsection_("stop", "Fine-tuning: convergence");
    registerIntOption_("stop:eval_every", "<n>", 1, "Validate every n epochs", false);
    registerIntOption_("stop:min_epochs", "<n>", 20, "Never stop before this epoch", false);
    registerIntOption_("stop:patience", "<n>", 10, "Stop after this many epochs without progress (never before max(min_epochs, warmup))", false);
    registerDoubleOption_("stop:rel_tol", "<f>", 0.005, "Progress = the selection metric beats the anchor by this fraction", false);
    registerDoubleOption_("stop:abs_tol", "<f>", 0.0, "Progress = beats the anchor by this absolute amount (0 = use rel_tol)", false);
    registerDoubleOption_("stop:max_seconds", "<s>", 0.0, "Wall-clock budget per head (0 = none)", false);
    registerStringOption_("stop:select", "<metric>", "calibrated_sd", "Selection metric on the validation cohort", false);
    setValidStrings_("stop:select", {"calibrated_sd", "rmse"});

    registerTOPPSubsection_("machine", "Fine-tuning: device");
    registerStringOption_("machine:device", "<dev>", "cpu", "cpu or cuda[:N]", false);
    registerIntOption_("machine:threads", "<n>", 4, "Torch threads on CPU (4 measured fastest on this model; more thrashes)", false);
    registerFlag_("machine:no_cudnn", "CUDA: do not use cuDNN (needed when only its loader shim is installed, as in pytorch.org's libtorch zips); slower");
    registerIntOption_("machine:seed", "<n>", 20260803, "Seed for the training subsample and batch order", false);
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

    const bool tune = getFlag_("tune");
#ifndef DLR_WITH_FINETUNE
    if (tune)
    {
      writeLogError_("this build has no fine-tuning stage (configured without DLR_BUILD_FINETUNE, which needs a "
                     "CXX11-ABI libtorch). Refinement itself is unaffected: drop -tune.");
      return ILLEGAL_PARAMETERS;
    }
#endif
    // minmax rescales RT over the MATCHED set. Re-predicting the whole library
    // first and then rescaling only the matched part leaves the two halves of
    // one column in different units, which no consumer can untangle afterwards.
    if (tune && p.rt_unit == ODIA::RefineParams::RtUnit::MinMax)
    {
      writeLogError_("-tune and -rt_unit minmax are incompatible: minmax rescales only the matched precursors, "
                     "and tuning exists to correct the unmatched ones");
      return ILLEGAL_PARAMETERS;
    }

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
    json tune_prov = json::object();
    try
    {
      ODIA::DIANNLibraryFile::load(in, library);
      writeLogInfo_("library: " + std::to_string(library.precursorCount()) + " precursors, " +
                    std::to_string(library.transitionCount()) + " transitions");

#ifdef DLR_WITH_FINETUNE
      if (tune)
      {
        namespace fs = std::filesystem;
        std::string models = getStringOption_("tune_models");
        if (models.empty())
        { if (const char* e = std::getenv("DIALIBGEN_MODEL_DIR"); e && *e) { models = e; } }
        if (models.empty())
        { throw std::runtime_error("-tune needs -tune:models (or $DIALIBGEN_MODEL_DIR): the stock peptdeep models to start from"); }

        const std::string heads = getStringOption_("tune_heads");
        const bool want_rt = heads != "ccs", want_ccs = heads != "rt";

        // Kept where asked, otherwise beside -out and removed at the end: the
        // tuned ONNX is an intermediate here, and the deliverable is the library.
        const std::string keep = getStringOption_("tune_out_models");
        const fs::path work = keep.empty() ? fs::path(out + ".tune") : fs::path(keep);
        fs::create_directories(work);

        ODIA::tune::TuneParams tp;
        tp.report = ids;
        tp.q_value = getDoubleOption_("filter:q_value");
        tp.min_charge = getIntOption_("filter:min_charge");
        tp.allow_z1 = getFlag_("filter:allow_z1");
        tp.rt_spread_max = getDoubleOption_("filter:rt_spread_max");
        tp.rt_max_minutes = getDoubleOption_("filter:rt_max_minutes");
        tp.train_size = static_cast<std::size_t>(std::max(0, getIntOption_("cohort:train_size")));
        tp.train_frac = getDoubleOption_("cohort:train_frac");
        tp.inner_val = !getFlag_("cohort:no_inner_val");
        tp.epochs = getIntOption_("train:epochs");
        tp.warmup = getIntOption_("train:warmup");
        tp.lr = getDoubleOption_("train:lr");
        tp.batch_size = getIntOption_("train:batch_size");
        tp.eval_every = getIntOption_("stop:eval_every");
        tp.min_epochs = getIntOption_("stop:min_epochs");
        tp.patience = getIntOption_("stop:patience");
        tp.rel_tol = getDoubleOption_("stop:rel_tol");
        tp.abs_tol = getDoubleOption_("stop:abs_tol");
        tp.max_seconds = getDoubleOption_("stop:max_seconds");
        tp.select = getStringOption_("stop:select") == "rmse" ? ODIA::tune::Select::Rmse : ODIA::tune::Select::CalibratedSd;
        tp.device = getStringOption_("machine:device");
        tp.threads = getIntOption_("machine:threads");
        tp.cudnn = !getFlag_("machine:no_cudnn");
        tp.seed = static_cast<std::uint32_t>(getIntOption_("machine:seed"));
        if (tp.epochs < 1 || tp.warmup < 0 || tp.warmup > tp.epochs)
        { throw std::runtime_error("train:warmup must be in [0, train:epochs]"); }
        if (tp.train_size && tp.train_frac > 0)
        { throw std::runtime_error("give cohort:train_size or cohort:train_frac, not both"); }

        auto run_head = [&](ODIA::tune::HeadKind head, const char* file)
        {
          tp.head = head;
          tp.model_in = (fs::path(models) / file).string();
          tp.model_out = (work / file).string();
          if (!fs::exists(tp.model_in))
          { throw std::runtime_error("no " + std::string(file) + " in " + models); }
          // finetune throws when nothing beat the stock model, the same way it
          // throws on a corrupt report -- TuneResult carries no "exported" flag
          // to tell them apart, so a head that does not improve aborts the run
          // rather than silently refining with stock predictions.
          return ODIA::tune::finetune(tp, std::cout);
        };

        const bool free_cys = getFlag_("tune_keep_free_cysteine_offset");
        const bool gpu = getFlag_("tune_predict_gpu");
        const unsigned sessions = static_cast<unsigned>(std::max(0, getIntOption_("tune_predict_sessions")));

        if (want_rt)
        {
          const ODIA::tune::TuneResult r = run_head(ODIA::tune::HeadKind::RT, "peptdeep_rt_dynamic.onnx");
          writeLogInfo_("tuned RT: " + r.stop_reason + ", best epoch " + std::to_string(r.best_epoch) +
                        " of " + std::to_string(r.epochs_run));
          // predictRetentionTimes returns what it could NOT predict, not what it
          // did -- undocumented, and it reads exactly the other way round.
          const std::size_t unpredicted =
            ODIA::LibraryGenerator::predictRetentionTimes(library, r.model_out, gpu, sessions, free_cys);
          const std::size_t n = library.precursorCount() - unpredicted;
          // The model emits rt_norm = RT / rt_max_minutes, and that denominator
          // lives nowhere else. Multiplying it back puts the WHOLE library in the
          // reference run's minutes -- the same unit refine() writes for the
          // matched precursors, so the column stays one coherent object.
          for (auto& v : library.precursors().irt) { v *= r.rt_max_minutes; }
          tune_prov["rt"] = {{"stop_reason", r.stop_reason}, {"best_epoch", r.best_epoch},
                             {"epochs_run", r.epochs_run}, {"rt_max_minutes", r.rt_max_minutes},
                             {"model_sha256", r.model_out_sha256}, {"stock_sha256", r.model_in_sha256},
                             {"repredicted", n}, {"unpredicted", unpredicted}};
          writeLogInfo_("re-predicted RT for " + std::to_string(n) + " of " + std::to_string(library.precursorCount()) +
                        " precursors, in the run's minutes" +
                        (unpredicted ? " (" + std::to_string(unpredicted) + " could not be encoded)" : ""));
        }
        if (want_ccs)
        {
          const ODIA::tune::TuneResult r = run_head(ODIA::tune::HeadKind::CCS, "peptdeep_ccs_dynamic.onnx");
          writeLogInfo_("tuned CCS: " + r.stop_reason + ", best epoch " + std::to_string(r.best_epoch) +
                        " of " + std::to_string(r.epochs_run));
          // derive_mobility rewrites the WHOLE 1/K0 column, so it runs only when
          // the CCS head actually tuned -- otherwise a library that arrived with
          // measured mobilities would lose them to stock predictions.
          const std::size_t unpredicted =
            ODIA::LibraryGenerator::predictCollisionCrossSections(library, r.model_out, gpu, sessions, true);
          const std::size_t n = library.precursorCount() - unpredicted;
          tune_prov["ccs"] = {{"stop_reason", r.stop_reason}, {"best_epoch", r.best_epoch},
                              {"epochs_run", r.epochs_run},
                              {"model_sha256", r.model_out_sha256}, {"stock_sha256", r.model_in_sha256},
                              {"repredicted", n}, {"unpredicted", unpredicted}};
          writeLogInfo_("re-predicted CCS and 1/K0 for " + std::to_string(n) + " of " +
                        std::to_string(library.precursorCount()) + " precursors" +
                        (unpredicted ? " (" + std::to_string(unpredicted) + " could not be encoded)" : ""));
        }

        tune_prov["models_kept"] = keep.empty() ? json(nullptr) : json(fs::absolute(work).string());
        if (keep.empty()) { std::error_code ec; fs::remove_all(work, ec); }
      }
#endif
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
      {"tune", tune_prov},
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
