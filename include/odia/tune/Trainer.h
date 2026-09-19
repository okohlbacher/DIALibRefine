// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// Fine-tune a stock peptdeep RT or CCS model on one run's DIA-NN report.
///
/// This is tools/finetune.py in C++ with libtorch, recipe unchanged:
///   L1 loss, Adam(lr 1e-4), clip_grad_norm 1.0, batches <= 1024 within
///   length groups (never padded), per-epoch LR = base * lambda(epoch) with
///   linear warmup then cosine to the horizon, dropout 0.1 in training.
/// Cohorts are protein-level and frozen before any subsampling: TEST is
/// crc32(protein group) % 5 == 0, the inner VALIDATION crc32("val:"+pg) % 7 == 0
/// of the rest, TRAINING a seeded random prefix of the remaining pool. The
/// stopping rule is the anchor-patience rule the sweep replays offline: a
/// checkpoint counts as progress only if it beats the anchor by rel_tol; after
/// `patience` non-progress checkpoints past `min_epochs` training stops and
/// the best-validation weights are restored and written back into the ONNX.
///
/// What is NOT in the model: the RT scale (rt_norm = RT / rt_max_minutes) and
/// the CCS <-> 1/K0 conversion (Mason-Schamp with ODIA's constants, per m/z
/// and charge). Both are applied here so every reported number is in the
/// unit a library carries (minutes, 1/K0).
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace ODIA::tune
{
  enum class HeadKind { RT, CCS };
  enum class Select { CalibratedSd, Rmse };

  struct TuneParams
  {
    HeadKind head = HeadKind::RT;
    std::string report;          ///< DIA-NN report.parquet (one run)
    std::string model_in;        ///< stock ONNX
    std::string model_out;       ///< tuned ONNX; sidecars <model_out>.tune.json / .trajectory.tsv
    // observation filter
    double q_value = 0.01;
    int min_charge = 2;          ///< CCS only: z1 is censored at the mobility ramp top; RT uses every charge
    bool allow_z1 = false;       ///< CCS: required to set min_charge below 2
    double rt_spread_max = 0.2;  ///< minutes; an RT unit whose charge states disagree by more is dropped
    double rt_max_minutes = 0;   ///< rt_norm denominator; 0 = max observed RT
    // cohorts
    std::size_t train_size = 0;  ///< 0 = full pool
    double train_frac = 0;       ///< alternative to train_size
    bool inner_val = true;       ///< false: validation = TEST (selection is then optimistic and TEST is no longer held out)
    /// Train on EVERY unit of the run, the test and validation cohorts included.
    ///
    /// This is fitting the run as closely as the data allows, and it destroys
    /// the tool's own generalisation numbers: val and TEST are then in-sample
    /// and only ever improve. It exists because the honest check for a per-run
    /// model is not a held-out protein cohort of the same run but a SEARCH OF A
    /// DIFFERENT RUN -- measured on K562 diaPASEF, a library tuned on its own run
    /// found 15,219 new precursors there and one tuned on a sibling run 19,996.
    /// Off by default, and the sidecar says when it was on.
    bool full_fit = false;
    // recipe
    int epochs = 100;
    int warmup = 10;
    double lr = 1e-4;
    int batch_size = 1024;
    // stopping
    int eval_every = 1;
    int min_epochs = 20;
    int patience = 10;           ///< epochs without progress (checkpoints when eval_every = 1)
    double rel_tol = 0.005;
    double abs_tol = 0;
    double max_seconds = 0;
    Select select = Select::CalibratedSd;
    // machine
    std::string device = "cpu";  ///< "cpu" or "cuda[:N]"
    bool cudnn = true;           ///< CUDA: let libtorch use cuDNN for the LSTM/conv (needs the full cuDNN 9 library set)
    int threads = 4;
    std::uint32_t seed = 20260803;
  };

  struct Metrics
  {
    std::size_t n = 0, nonfinite_predictions = 0;
    double rmse = NAN, sd = NAN, mean_err = NAN, p95 = NAN;
    double calibrated_sd = NAN, cal_slope = NAN, cal_intercept = NAN;
    std::map<int, double> sd_by_charge;        ///< CCS only, groups of >= 30
    std::map<int, std::size_t> n_by_charge;
    std::map<int, double> mean_by_charge;
  };

  struct TuneResult
  {
    // cohorts
    std::size_t observations = 0, units = 0, test = 0, val = 0, pool = 0, training = 0;
    bool val_is_test = false;
    double rt_max_minutes = 0;
    // course
    int epochs_run = 0, best_epoch = 0;
    std::size_t updates = 0;
    double train_seconds = 0, eval_seconds = 0;
    std::string stop_reason;
    double param_l2_change = 0;
    // evaluation, deployed units
    Metrics stock_val, stock_test, tuned_val, tuned_test;
    std::string model_out, model_in_sha256, model_out_sha256;
  };

  /// Run the fine-tune. Throws std::runtime_error with a reason a user can act
  /// on -- including when no checkpoint beat the stock model, in which case the
  /// provenance sidecar is written and the model is not.
  TuneResult finetune(const TuneParams& params, std::ostream& log);

  const char* headName(HeadKind h);
  const char* selectName(Select s);
}
