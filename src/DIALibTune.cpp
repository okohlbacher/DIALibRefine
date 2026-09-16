// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// DIALibTune: fine-tune a peptdeep RT or CCS ONNX model on one DIA-NN run,
/// in C++ with libtorch, and write the tuned model back as the same ONNX file
/// (byte layout preserved -- only the weights change), ready for DIALibGen.

#include <odia/tune/Trainer.h>

#include "ToolBoilerplate.h"

#include <OpenMS/APPLICATIONS/TOPPBase.h>
#include <OpenMS/CONCEPT/LogStream.h>

#include <iostream>

using namespace OpenMS;
using namespace ODIA::tune;

class TOPPDIALibTune : public TOPPBase
{
public:
  TOPPDIALibTune() :
    TOPPBase("DIALibTune",
             "Fine-tune a peptdeep RT or CCS model on one DIA-NN run (libtorch)",
             false, {}, false)
  {
#ifdef DLR_VERSION
    // Our own version, not the OpenMS this happened to be built against.
    version_ = DLR_VERSION;
    verboseVersion_ = dlr::verboseVersion();
#endif
  }

protected:
  void registerOptionsAndFlags_() override
  {
    registerInputFile_("in", "<file>", "", "DIA-NN report.parquet of ONE run");
    setValidFormats_("in", {"parquet"});
    registerInputFile_("model_in", "<file>", "", "Stock peptdeep ONNX (rt.onnx or ccs.onnx)");
    registerOutputFile_("out", "<file>", "", "Tuned ONNX. Sidecars: <out>.tune.json (provenance, metrics) and <out>.trajectory.tsv (per-epoch)");
    registerStringOption_("head", "<rt|ccs>", "rt", "Which model the ONNX is", false);
    setValidStrings_("head", {"rt", "ccs"});

    registerTOPPSubsection_("filter", "Observation filter");
    registerDoubleOption_("filter:q_value", "<q>", 0.01, "Precursor Q.Value threshold", false);
    registerIntOption_("filter:min_charge", "<z>", 2, "CCS only: lowest precursor charge used (RT collapses all charges). z1 is censored at the mobility ramp top on timsTOF", false);
    setMinInt_("filter:min_charge", 1); setMaxInt_("filter:min_charge", 8);
    registerFlag_("filter:allow_z1", "CCS: permit filter:min_charge 1 (censored observations enter training)");
    registerDoubleOption_("filter:rt_spread_max", "<min>", 0.2, "Drop a sequence whose charge states' RTs span more than this (minutes)", false);
    registerDoubleOption_("filter:rt_max_minutes", "<min>", 0.0, "rt_norm denominator; 0 = the run's maximum observed RT", false);

    registerTOPPSubsection_("cohort", "Protein-level cohorts (frozen before subsampling)");
    registerIntOption_("cohort:train_size", "<n>", 0, "Training units (sequences for rt, sequence x charge for ccs); 0 = full pool", false);
    registerDoubleOption_("cohort:train_frac", "<f>", 0.0, "Alternative to train_size: fraction of the pool", false);
    registerFlag_("cohort:no_inner_val", "No inner validation cohort: its units rejoin the pool and checkpoints are selected on TEST, which is then no longer a held-out number");

    registerTOPPSubsection_("train", "Recipe");
    registerIntOption_("train:epochs", "<n>", 100, "Horizon of the cosine schedule (and the maximum epochs)", false);
    registerIntOption_("train:warmup", "<n>", 10, "Linear warmup epochs", false);
    registerDoubleOption_("train:lr", "<lr>", 1e-4, "Peak learning rate (Adam)", false);
    registerIntOption_("train:batch_size", "<n>", 1024, "Batch size within a length group", false);

    registerTOPPSubsection_("stop", "Convergence");
    registerIntOption_("stop:eval_every", "<n>", 1, "Validate every n epochs", false);
    registerIntOption_("stop:min_epochs", "<n>", 20, "Never stop before this epoch", false);
    registerIntOption_("stop:patience", "<n>", 10, "Stop after this many epochs without progress (never before max(min_epochs, warmup))", false);
    registerDoubleOption_("stop:rel_tol", "<f>", 0.005, "Progress = the selection metric beats the anchor by this fraction", false);
    registerDoubleOption_("stop:abs_tol", "<f>", 0.0, "Progress = beats the anchor by this absolute amount (0 = use rel_tol)", false);
    registerDoubleOption_("stop:max_seconds", "<s>", 0.0, "Wall-clock budget for training (0 = none)", false);
    registerStringOption_("stop:select", "<metric>", "calibrated_sd", "Selection metric on the validation cohort", false);
    setValidStrings_("stop:select", {"calibrated_sd", "rmse"});

    registerTOPPSubsection_("machine", "Device");
    registerStringOption_("machine:device", "<dev>", "cpu", "cpu or cuda[:N]", false);
    registerIntOption_("machine:threads", "<n>", 4, "Torch threads on CPU (4 measured fastest on this model; more thrashes)", false);
    registerIntOption_("machine:seed", "<n>", 20260803, "Seed for the training subsample and batch order", false);
  }

  ExitCodes main_(int, const char**) override
  {
    TuneParams p;
    p.head = getStringOption_("head") == "ccs" ? HeadKind::CCS : HeadKind::RT;
    p.report = getStringOption_("in");
    p.model_in = getStringOption_("model_in");
    p.model_out = getStringOption_("out");
    p.q_value = getDoubleOption_("filter:q_value");
    p.min_charge = getIntOption_("filter:min_charge");
    p.allow_z1 = getFlag_("filter:allow_z1");
    p.rt_spread_max = getDoubleOption_("filter:rt_spread_max");
    p.rt_max_minutes = getDoubleOption_("filter:rt_max_minutes");
    const int train_size = getIntOption_("cohort:train_size");
    if (train_size < 0) { writeLogError_("cohort:train_size must be >= 0"); return ILLEGAL_PARAMETERS; }
    p.train_size = static_cast<std::size_t>(train_size);
    p.train_frac = getDoubleOption_("cohort:train_frac");
    p.inner_val = !getFlag_("cohort:no_inner_val");
    p.epochs = getIntOption_("train:epochs");
    p.warmup = getIntOption_("train:warmup");
    p.lr = getDoubleOption_("train:lr");
    p.batch_size = getIntOption_("train:batch_size");
    p.eval_every = getIntOption_("stop:eval_every");
    p.min_epochs = getIntOption_("stop:min_epochs");
    p.patience = getIntOption_("stop:patience");
    p.rel_tol = getDoubleOption_("stop:rel_tol");
    p.abs_tol = getDoubleOption_("stop:abs_tol");
    p.max_seconds = getDoubleOption_("stop:max_seconds");
    p.select = getStringOption_("stop:select") == "rmse" ? Select::Rmse : Select::CalibratedSd;
    p.device = getStringOption_("machine:device");
    p.threads = getIntOption_("machine:threads");
    p.seed = static_cast<std::uint32_t>(getIntOption_("machine:seed"));

    if (p.epochs < 1 || p.warmup < 0 || p.warmup > p.epochs) { writeLogError_("train:warmup must be in [0, train:epochs]"); return ILLEGAL_PARAMETERS; }
    if (p.batch_size < 1 || p.eval_every < 1 || p.patience < 1 || p.min_epochs < 0) { writeLogError_("batch_size, eval_every, patience must be >= 1 and min_epochs >= 0"); return ILLEGAL_PARAMETERS; }
    if (p.train_size && p.train_frac > 0) { writeLogError_("give cohort:train_size or cohort:train_frac, not both"); return ILLEGAL_PARAMETERS; }
    if (p.train_frac < 0 || p.train_frac > 1) { writeLogError_("cohort:train_frac must be in [0,1]"); return ILLEGAL_PARAMETERS; }
    if (p.rel_tol < 0 || p.abs_tol < 0 || p.max_seconds < 0 || p.lr <= 0 || p.q_value < 0) { writeLogError_("rel_tol, abs_tol, max_seconds and q_value must be >= 0 and lr > 0"); return ILLEGAL_PARAMETERS; }
    if (p.batch_size > 1024) { OPENMS_LOG_WARN << "train:batch_size " << p.batch_size << " exceeds the recipe's 1024; the measured results used 1024\n"; }

    try
    {
      const TuneResult r = finetune(p, OPENMS_LOG_INFO);
      OPENMS_LOG_INFO << "wrote " << r.model_out << "  (" << r.stop_reason << ", best epoch " << r.best_epoch
                      << " of " << r.epochs_run << ", " << r.updates << " updates, " << r.train_seconds << " s training)\n";
    }
    catch (const std::exception& e)
    {
      writeLogError_(std::string("DIALibTune: ") + e.what());
      return INPUT_FILE_CORRUPT;
    }
    return EXECUTION_OK;
  }
};

int main(int argc, const char** argv)
{
  dlr::disableUpdateCheckUnlessSet();
  dlr::ToolHandlerRegistration ttd("DIALibTune");
  TOPPDIALibTune tool;
  return tool.main(argc, argv);
}
