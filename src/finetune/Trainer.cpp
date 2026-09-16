// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <odia/tune/Trainer.h>
#include <odia/tune/PeptDeepModel.h>
#include <odia/tune/OnnxWeights.h>

#include <odia/DIANNLibraryFile.h>
#include <odia/Library.h>
#include <odia/PeptDeepEncoder.h>

#include <OpenMS/CHEMISTRY/AASequence.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace ODIA::tune
{
  const char* headName(HeadKind h) { return h == HeadKind::CCS ? "ccs" : "rt"; }
  const char* selectName(Select s) { return s == Select::Rmse ? "rmse" : "calibrated_sd"; }

  namespace
  {
    using Clock = std::chrono::steady_clock;
    double seconds(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

    // ---- Parquet -------------------------------------------------------------

    std::shared_ptr<arrow::ChunkedArray> column(const std::shared_ptr<arrow::Table>& t, const char* name, bool required = true)
    {
      auto c = t->GetColumnByName(name);
      if (!c && required) { throw std::runtime_error(std::string("report lacks column ") + name); }
      return c;
    }

    std::vector<double> toDoubles(const std::shared_ptr<arrow::ChunkedArray>& c)
    {
      auto casted = arrow::compute::Cast(c, arrow::float64());
      if (!casted.ok()) { throw std::runtime_error("cannot read a numeric column: " + casted.status().ToString()); }
      std::vector<double> out; out.reserve(static_cast<std::size_t>(c->length()));
      for (const auto& chunk : casted->chunked_array()->chunks())
      {
        auto a = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        for (std::int64_t i = 0; i < a->length(); ++i) { out.push_back(a->IsNull(i) ? NAN : a->Value(i)); }
      }
      return out;
    }

    std::vector<std::string> toStrings(const std::shared_ptr<arrow::ChunkedArray>& c)
    {
      auto casted = arrow::compute::Cast(c, arrow::utf8());
      if (!casted.ok()) { throw std::runtime_error("cannot read a string column: " + casted.status().ToString()); }
      std::vector<std::string> out; out.reserve(static_cast<std::size_t>(c->length()));
      for (const auto& chunk : casted->chunked_array()->chunks())
      {
        auto a = std::static_pointer_cast<arrow::StringArray>(chunk);
        for (std::int64_t i = 0; i < a->length(); ++i) { out.push_back(a->IsNull(i) ? std::string() : a->GetString(i)); }
      }
      return out;
    }

    std::shared_ptr<arrow::Table> readParquet(const std::string& path)
    {
      auto infile = arrow::io::ReadableFile::Open(path);
      if (!infile.ok()) { throw std::runtime_error("cannot open " + path + ": " + infile.status().ToString()); }
      auto reader = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
      if (!reader.ok()) { throw std::runtime_error("not a Parquet file: " + path + ": " + reader.status().ToString()); }
      std::shared_ptr<arrow::Table> table;
      const auto st = (*reader)->ReadTable(&table);
      if (!st.ok()) { throw std::runtime_error("cannot read " + path + ": " + st.ToString()); }
      return table;
    }

    // ---- cohorts: the same hash as tools/finetune.py (zlib.crc32) -------------

    std::uint32_t crc32(const std::string& s)
    {
      static const auto table = []
      {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i)
        {
          std::uint32_t c = i;
          for (int k = 0; k < 8; ++k) { c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
          t[i] = c;
        }
        return t;
      }();
      std::uint32_t c = 0xFFFFFFFFu;
      for (unsigned char ch : s) { c = table[(c ^ ch) & 0xFF] ^ (c >> 8); }
      return c ^ 0xFFFFFFFFu;
    }

    double median(std::vector<double> v)
    {
      std::sort(v.begin(), v.end());
      const std::size_t n = v.size();
      return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    // ---- data ------------------------------------------------------------------

    /// One training unit: a modified sequence (RT) or a sequence x charge (CCS).
    struct Unit
    {
      OpenMS::AASequence peptide;
      int charge = 0;
      double mz = 0;
      double observed = 0;   ///< minutes (RT) or 1/K0 (CCS)
      double target = 0;     ///< rt_norm or CCS in A^2 -- what the model predicts
      std::string protein_group;
      enum Cohort : std::uint8_t { Test, Val, Pool } cohort = Pool;
      bool training = false;
    };

    struct Rejections
    {
      std::size_t q = 0, charge = 0, decoy = 0, nonfinite = 0, unparsable = 0, unencodable = 0, spread = 0;
      nlohmann::json json() const
      { return {{"q_value", q}, {"charge", charge}, {"decoy", decoy}, {"nonfinite", nonfinite}, {"unparsable_sequence", unparsable}, {"unencodable_sequence", unencodable}, {"rt_spread", spread}}; }
    };

    struct Dataset
    {
      std::vector<Unit> units;
      std::size_t observations = 0, rows = 0;
      std::string run;
      double rt_max_minutes = 0;
      Rejections rejected;
    };

    Dataset loadReport(const TuneParams& p, std::ostream& log)
    {
      auto table = readParquet(p.report);
      Dataset d;
      d.rows = static_cast<std::size_t>(table->num_rows());
      const auto seq = toStrings(column(table, "Modified.Sequence"));
      const auto pg = toStrings(column(table, "Protein.Group"));
      const auto run = toStrings(column(table, "Run"));
      const auto z = toDoubles(column(table, "Precursor.Charge"));
      const auto mz = toDoubles(column(table, "Precursor.Mz"));
      const auto rt = toDoubles(column(table, "RT"));
      const auto im = toDoubles(column(table, "IM"));
      const auto q = toDoubles(column(table, "Q.Value"));
      auto decoy_col = column(table, "Decoy", false);
      const auto decoy = decoy_col ? toDoubles(decoy_col) : std::vector<double>();

      std::set<std::string> runs(run.begin(), run.end());
      if (runs.size() != 1) { throw std::runtime_error("the report holds " + std::to_string(runs.size()) + " runs; fine-tuning is per run -- give a single-run report"); }
      d.run = *runs.begin();

      const bool ccs = p.head == HeadKind::CCS;
      // The charge floor is a CCS matter (z1 is censored at the ramp top); RT
      // units collapse charge states anyway, so every charge contributes there.
      const int z_min = ccs ? p.min_charge : 1;
      // key -> observations
      struct Obs { double rt, im, mz; std::string pg; };
      std::map<std::pair<std::string, int>, std::vector<Obs>> by_key;
      for (std::size_t i = 0; i < d.rows; ++i)
      {
        if (!decoy.empty() && decoy[i] == 1) { ++d.rejected.decoy; continue; }
        if (!(q[i] <= p.q_value)) { ++d.rejected.q; continue; }
        const int zi = static_cast<int>(z[i]);
        if (!std::isfinite(z[i]) || z[i] != zi || zi < z_min) { ++d.rejected.charge; continue; }
        const double v = ccs ? im[i] : rt[i];
        // finite RT >= 0, finite IM > 0, finite m/z > 0 -- the Python contract
        if (!std::isfinite(v) || (ccs ? v <= 0 : v < 0) || !std::isfinite(mz[i]) || mz[i] <= 0) { ++d.rejected.nonfinite; continue; }
        by_key[{seq[i], ccs ? zi : 0}].push_back({rt[i], im[i], mz[i], pg[i]});
        ++d.observations;
      }

      double rt_hi = 0;
      for (auto& [key, obs] : by_key)
      {
        Unit u;
        try { u.peptide = OpenMS::AASequence::fromString(key.first); }
        catch (const std::exception&) { ++d.rejected.unparsable; continue; }
        try { (void)PeptDeepEncoder::encode(u.peptide); }
        catch (const std::exception&) { ++d.rejected.unencodable; continue; }
        std::vector<double> vals, mzs;
        for (const auto& o : obs) { vals.push_back(ccs ? o.im : o.rt); mzs.push_back(o.mz); }
        if (!ccs && obs.size() > 1)
        {
          const auto [lo, hi] = std::minmax_element(vals.begin(), vals.end());
          if (*hi - *lo > p.rt_spread_max) { ++d.rejected.spread; continue; }
        }
        u.charge = key.second;
        u.mz = median(mzs);
        u.observed = median(vals);
        u.protein_group = obs.front().pg;
        if (!ccs) { rt_hi = std::max(rt_hi, u.observed); }
        d.units.push_back(std::move(u));
      }
      if (d.units.empty()) { throw std::runtime_error("no usable observations after filtering (q <= " + std::to_string(p.q_value) + ", charge >= " + std::to_string(p.min_charge) + ")"); }

      d.rt_max_minutes = p.rt_max_minutes > 0 ? p.rt_max_minutes : rt_hi;
      for (auto& u : d.units)
      {
        u.target = ccs ? ccsFromMobility(u.observed, u.mz, u.charge) : u.observed / d.rt_max_minutes;
        u.cohort = crc32(u.protein_group) % 5 == 0 ? Unit::Test
                 : crc32("val:" + u.protein_group) % 7 == 0 ? Unit::Val : Unit::Pool;
      }
      log << "report: " << d.rows << " rows, " << d.observations << " observations (q <= " << p.q_value << ", charge >= " << z_min << "), " << d.units.size() << " "
          << (ccs ? "sequence x charge" : "sequence") << " units (rejected: q " << d.rejected.q << ", charge " << d.rejected.charge
          << ", decoy " << d.rejected.decoy << ", non-finite " << d.rejected.nonfinite << ", unparsable " << d.rejected.unparsable
          << ", unencodable " << d.rejected.unencodable << ", rt spread " << d.rejected.spread << ")\n";
      return d;
    }

    // ---- encoded tensors, one block per peptide length ------------------------

    struct Block
    {
      std::vector<std::size_t> unit;          ///< unit index per row
      torch::Tensor aa, mod_x, charges, target;
      std::vector<std::int64_t> train_rows, val_rows, test_rows;
    };

    std::vector<Block> encode(const std::vector<Unit>& units, const torch::Device& dev)
    {
      std::vector<OpenMS::AASequence> peps; peps.reserve(units.size());
      for (const auto& u : units) { peps.push_back(u.peptide); }
      std::vector<Block> blocks;
      for (const auto& group : PeptDeepEncoder::groupByLength(peps))
      {
        std::vector<OpenMS::AASequence> gp; std::vector<int> gz; std::vector<float> gy;
        for (auto i : group) { gp.push_back(units[i].peptide); gz.push_back(std::max(1, units[i].charge)); gy.push_back(static_cast<float>(units[i].target)); }
        auto b = PeptDeepEncoder::encode(gp, gz, 30.0f, "Lumos");
        const auto rows = static_cast<std::int64_t>(b.rows), L = static_cast<std::int64_t>(b.sequence_length);
        Block blk;
        blk.unit = group;
        blk.aa = torch::from_blob(b.aa_indices.data(), {rows, L}, torch::kInt64).clone().to(dev);
        blk.mod_x = torch::from_blob(b.mod_x.data(), {rows, L, MOD_FEATURES}, torch::kFloat32).clone().to(dev);
        blk.charges = torch::from_blob(b.charges.data(), {rows, 1}, torch::kFloat32).clone().to(dev);
        blk.target = torch::from_blob(gy.data(), {rows}, torch::kFloat32).clone().to(dev);
        for (std::int64_t r = 0; r < rows; ++r)
        {
          const auto& u = units[group[static_cast<std::size_t>(r)]];
          if (u.cohort == Unit::Test) { blk.test_rows.push_back(r); }
          else if (u.cohort == Unit::Val) { blk.val_rows.push_back(r); }
          if (u.training) { blk.train_rows.push_back(r); }
        }
        blocks.push_back(std::move(blk));
      }
      return blocks;
    }

    torch::Tensor rowsTensor(const std::vector<std::int64_t>& rows, std::size_t from, std::size_t to, const torch::Device& dev)
    {
      return torch::from_blob(const_cast<std::int64_t*>(rows.data()) + from, {static_cast<std::int64_t>(to - from)}, torch::kInt64).clone().to(dev);
    }

    // ---- evaluation, deployed units ----------------------------------------------

    /// Predictions for one cohort of every block, written into pred[unit].
    void predict(Head& model, const std::vector<Block>& blocks, Unit::Cohort which, int batch, const torch::Device& dev, std::vector<double>& pred)
    {
      torch::NoGradGuard ng;
      model->eval();
      for (const auto& b : blocks)
      {
        const auto& rows = which == Unit::Test ? b.test_rows : b.val_rows;
        for (std::size_t s = 0; s < rows.size(); s += static_cast<std::size_t>(batch))
        {
          const std::size_t e = std::min(rows.size(), s + static_cast<std::size_t>(batch));
          auto idx = rowsTensor(rows, s, e, dev);
          auto y = model->forward(b.aa.index_select(0, idx), b.mod_x.index_select(0, idx), b.charges.index_select(0, idx)).to(torch::kCPU).contiguous();
          const float* yp = y.data_ptr<float>();
          for (std::size_t k = s; k < e; ++k) { pred[b.unit[static_cast<std::size_t>(rows[k])]] = yp[k - s]; }
        }
      }
    }

    double sd(const std::vector<double>& r)
    {
      if (r.size() < 2) { return NAN; }
      const double m = std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
      double ss = 0; for (double x : r) { ss += (x - m) * (x - m); }
      return std::sqrt(ss / static_cast<double>(r.size() - 1));
    }

    Metrics metrics(const Dataset& d, const std::vector<double>& pred, Unit::Cohort which, bool ccs)
    {
      std::vector<double> obs, dep;             // deployed units: minutes or 1/K0
      std::map<int, std::vector<double>> by_z;
      for (std::size_t i = 0; i < d.units.size(); ++i)
      {
        const auto& u = d.units[i];
        if (u.cohort != which || !std::isfinite(pred[i])) { continue; }
        const double p = ccs ? mobilityFromCCS(pred[i], u.mz, u.charge) : pred[i] * d.rt_max_minutes;
        obs.push_back(u.observed); dep.push_back(p);
        if (ccs) { by_z[u.charge].push_back(p - u.observed); }
      }
      Metrics m; m.n = obs.size();
      if (m.n < 2) { return m; }
      std::vector<double> r(m.n), ar(m.n);
      double ss = 0;
      for (std::size_t i = 0; i < m.n; ++i) { r[i] = dep[i] - obs[i]; ar[i] = std::fabs(r[i]); ss += r[i] * r[i]; }
      m.rmse = std::sqrt(ss / static_cast<double>(m.n));
      m.sd = sd(r);
      m.mean_err = std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(m.n);
      std::sort(ar.begin(), ar.end());
      {  // numpy.percentile default: linear interpolation between order statistics
        const double pos = 0.95 * static_cast<double>(m.n - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const double frac = pos - static_cast<double>(lo);
        m.p95 = lo + 1 < m.n ? ar[lo] + frac * (ar[lo + 1] - ar[lo]) : ar[lo];
      }
      // least squares obs ~ a * pred + b; calibrated sd = sd of the fit's residual
      const double n = static_cast<double>(m.n);
      const double mx = std::accumulate(dep.begin(), dep.end(), 0.0) / n, my = std::accumulate(obs.begin(), obs.end(), 0.0) / n;
      double sxx = 0, sxy = 0;
      for (std::size_t i = 0; i < m.n; ++i) { sxx += (dep[i] - mx) * (dep[i] - mx); sxy += (dep[i] - mx) * (obs[i] - my); }
      m.cal_slope = sxx > 0 ? sxy / sxx : NAN;
      m.cal_intercept = my - m.cal_slope * mx;
      std::vector<double> cr(m.n);
      for (std::size_t i = 0; i < m.n; ++i) { cr[i] = obs[i] - (m.cal_slope * dep[i] + m.cal_intercept); }
      m.calibrated_sd = sd(cr);
      for (auto& [zc, v] : by_z) { m.sd_by_charge[zc] = sd(v); }
      return m;
    }

    nlohmann::json json(const Metrics& m)
    {
      auto f = [](double x) { return std::isfinite(x) ? nlohmann::json(x) : nlohmann::json(nullptr); };
      nlohmann::json j = {{"n", m.n}, {"rmse", f(m.rmse)}, {"sd", f(m.sd)}, {"mean_err", f(m.mean_err)}, {"p95", f(m.p95)},
                          {"calibrated_sd", f(m.calibrated_sd)}, {"cal_slope", f(m.cal_slope)}, {"cal_intercept", f(m.cal_intercept)}};
      if (!m.sd_by_charge.empty()) { for (auto& [z, s] : m.sd_by_charge) { j["sd_by_charge"][std::to_string(z)] = f(s); } }
      return j;
    }

    double selected(const Metrics& m, Select s) { return s == Select::Rmse ? m.rmse : m.calibrated_sd; }

    double lrLambda(int epoch, int warmup, int horizon)
    {
      if (epoch < warmup) { return static_cast<double>(epoch + 1) / static_cast<double>(warmup); }
      if (horizon <= warmup) { return 1.0; }
      const double t = static_cast<double>(epoch - warmup) / static_cast<double>(horizon - warmup);
      return std::max(1e-10, 0.5 * (1.0 + std::cos(M_PI * t)));
    }

    std::vector<torch::Tensor> snapshot(Head& model)
    {
      torch::NoGradGuard ng;
      std::vector<torch::Tensor> s;
      for (auto& p : model->parameters()) { s.push_back(p.detach().clone()); }
      return s;
    }
    void restore(Head& model, const std::vector<torch::Tensor>& s)
    {
      torch::NoGradGuard ng;
      std::size_t i = 0;
      for (auto& p : model->parameters()) { p.copy_(s[i++]); }
    }
    double l2Change(Head& model, const std::vector<torch::Tensor>& s)
    {
      torch::NoGradGuard ng;
      double acc = 0; std::size_t i = 0;
      for (auto& p : model->parameters()) { acc += (p - s[i++]).pow(2).sum().item<double>(); }
      return std::sqrt(acc);
    }
  }

  TuneResult finetune(const TuneParams& p, std::ostream& log)
  {
    const bool ccs = p.head == HeadKind::CCS;
    TuneResult res;
    res.model_out = p.model_out;

    // device
    torch::Device dev(torch::kCPU);
    if (p.device.rfind("cuda", 0) == 0)
    {
      if (!torch::cuda::is_available()) { throw std::runtime_error("device " + p.device + " requested but CUDA is not available to this libtorch (no driver, no GPU, or a CPU-only build)"); }
      dev = torch::Device(p.device);
    }
    else if (p.device != "cpu") { throw std::runtime_error("unknown device " + p.device); }
    if (dev.is_cpu()) { torch::set_num_threads(std::max(1, p.threads)); }
    torch::manual_seed(p.seed);
    log << "device " << p.device << (dev.is_cpu() ? " (" + std::to_string(torch::get_num_threads()) + " threads)" : "") << ", libtorch " << TORCH_VERSION << "\n";

    // data + cohorts
    Dataset d = loadReport(p, log);
    std::vector<std::size_t> pool;
    for (std::size_t i = 0; i < d.units.size(); ++i)
    {
      if (d.units[i].cohort == Unit::Test) { ++res.test; }
      else if (d.units[i].cohort == Unit::Val) { ++res.val; }
      else { pool.push_back(i); }
    }
    res.pool = pool.size();
    res.units = d.units.size();
    res.observations = d.observations;
    res.rt_max_minutes = d.rt_max_minutes;
    std::mt19937 rng(p.seed);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::size_t n_train = pool.size();
    if (p.train_size) { n_train = std::min(n_train, p.train_size); }
    else if (p.train_frac > 0) { n_train = static_cast<std::size_t>(std::floor(p.train_frac * static_cast<double>(pool.size()))); }
    if (n_train == 0) { throw std::runtime_error("training set is empty"); }
    for (std::size_t k = 0; k < n_train; ++k) { d.units[pool[k]].training = true; }
    res.training = n_train;
    if (!p.inner_val)
    {
      // validation = training: every val unit rejoins the pool; selection becomes optimistic
      for (auto& u : d.units) { if (u.cohort == Unit::Val) { u.cohort = Unit::Pool; } }
      for (auto& u : d.units) { if (u.training) { u.cohort = Unit::Val; } }
      res.val = n_train; res.pool += 0;
    }
    if (res.val < 2 || res.test < 2) { throw std::runtime_error("validation or test cohort has fewer than 2 units (val " + std::to_string(res.val) + ", test " + std::to_string(res.test) + ")"); }
    log << "cohorts: test " << res.test << " (protein-held-out), val " << res.val << (p.inner_val ? " (protein-held-out)" : " (= training)")
        << ", pool " << res.pool << ", training " << res.training << (p.train_size || p.train_frac > 0 ? " (subsample, seed " + std::to_string(p.seed) + ")" : " (full pool)")
        << (ccs ? "" : ", rt_norm = RT / " + std::to_string(d.rt_max_minutes)) << "\n";

    // model
    OnnxFile onnx = OnnxFile::read(p.model_in);
    res.model_in_sha256 = DIANNLibraryFile::hashFile(p.model_in);
    Head model(ccs);
    loadWeights(onnx, model);
    model->to(dev);
    const auto stock = snapshot(model);

    auto t_enc = Clock::now();
    auto blocks = encode(d.units, dev);
    std::size_t train_rows = 0; for (const auto& b : blocks) { train_rows += b.train_rows.size(); }
    log << "encoded " << blocks.size() << " length groups in " << seconds(t_enc) << " s\n";

    std::vector<double> pred(d.units.size(), NAN);
    auto t_eval = Clock::now();
    predict(model, blocks, Unit::Val, p.batch_size, dev, pred);
    predict(model, blocks, Unit::Test, p.batch_size, dev, pred);
    res.stock_val = metrics(d, pred, Unit::Val, ccs);
    res.stock_test = metrics(d, pred, Unit::Test, ccs);
    res.eval_seconds += seconds(t_eval);
    const char* unit = ccs ? "1/K0" : "min";
    log << "stock: val " << selectName(p.select) << " " << selected(res.stock_val, p.select) << " " << unit
        << ", TEST sd " << res.stock_test.sd << " calibrated " << res.stock_test.calibrated_sd << " " << unit << "\n";

    // optimizer: Adam on the trainable parameters only (h0/c0 stay frozen)
    auto params = model->trainable();
    torch::optim::Adam opt(params, torch::optim::AdamOptions(p.lr).betas({0.9, 0.999}).eps(1e-8));
    auto setLr = [&](double lr) { for (auto& g : opt.param_groups()) { static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr); } };

    std::ofstream traj(p.model_out + ".trajectory.tsv");
    traj << "epoch\tlr\tupdates\ttrain_loss\tval_rmse\tval_sd\tval_mean_err\tval_p95\tval_calibrated_sd\ttrain_s\teval_s\tbest\tstop\n";

    double best = INFINITY, anchor = NAN; int stale = 0;
    std::vector<torch::Tensor> best_state = stock;
    res.best_epoch = 0;
    std::string stop;
    std::vector<std::size_t> block_order(blocks.size());
    std::iota(block_order.begin(), block_order.end(), 0);

    for (int epoch = 0; epoch < p.epochs; ++epoch)
    {
      const double lr = p.lr * lrLambda(epoch, p.warmup, p.epochs);
      setLr(lr);
      model->train();
      auto t0 = Clock::now();
      double loss_sum = 0; std::size_t loss_n = 0;
      std::shuffle(block_order.begin(), block_order.end(), rng);
      for (auto bi : block_order)
      {
        auto& b = blocks[bi];
        if (b.train_rows.empty()) { continue; }
        std::shuffle(b.train_rows.begin(), b.train_rows.end(), rng);
        for (std::size_t s = 0; s < b.train_rows.size(); s += static_cast<std::size_t>(p.batch_size))
        {
          const std::size_t e = std::min(b.train_rows.size(), s + static_cast<std::size_t>(p.batch_size));
          auto idx = rowsTensor(b.train_rows, s, e, dev);
          opt.zero_grad();
          auto y = model->forward(b.aa.index_select(0, idx), b.mod_x.index_select(0, idx), b.charges.index_select(0, idx));
          auto loss = torch::l1_loss(y, b.target.index_select(0, idx));
          loss.backward();
          torch::nn::utils::clip_grad_norm_(params, 1.0);
          opt.step();
          ++res.updates;
          const double lv = loss.item<double>();
          if (!std::isfinite(lv)) { throw std::runtime_error("non-finite loss at epoch " + std::to_string(epoch + 1) + " -- training diverged"); }
          loss_sum += lv * static_cast<double>(e - s); loss_n += e - s;
        }
      }
      res.train_seconds += seconds(t0);
      res.epochs_run = epoch + 1;
      const double train_loss = loss_n ? loss_sum / static_cast<double>(loss_n) : NAN;

      bool evaluated = false; Metrics vm; bool is_best = false;
      if ((epoch + 1) % p.eval_every == 0 || epoch + 1 == p.epochs)
      {
        auto t1 = Clock::now();
        predict(model, blocks, Unit::Val, p.batch_size, dev, pred);
        vm = metrics(d, pred, Unit::Val, ccs);
        res.eval_seconds += seconds(t1);
        evaluated = true;
        const double m = selected(vm, p.select);
        if (!std::isfinite(m)) { throw std::runtime_error("non-finite validation metric at epoch " + std::to_string(epoch + 1)); }
        if (m < best) { best = m; res.best_epoch = epoch + 1; best_state = snapshot(model); is_best = true; }
        const bool progress = !std::isfinite(anchor) || (p.abs_tol > 0 ? anchor - m >= p.abs_tol : anchor - m >= p.rel_tol * anchor);
        if (progress) { anchor = m; stale = 0; } else { ++stale; }
        if (epoch + 1 >= p.min_epochs && stale >= p.patience) { stop = "patience (" + std::to_string(p.patience) + " checkpoints without " + (p.abs_tol > 0 ? std::to_string(p.abs_tol) + " absolute" : std::to_string(100 * p.rel_tol) + "% relative") + " progress)"; }
      }
      if (stop.empty() && p.max_seconds > 0 && res.train_seconds >= p.max_seconds) { stop = "max_seconds"; }
      if (stop.empty() && epoch + 1 == p.epochs) { stop = "horizon"; }

      traj << epoch + 1 << '\t' << lr << '\t' << res.updates << '\t' << train_loss << '\t';
      if (evaluated) { traj << vm.rmse << '\t' << vm.sd << '\t' << vm.mean_err << '\t' << vm.p95 << '\t' << vm.calibrated_sd; }
      else { traj << "\t\t\t\t"; }
      traj << '\t' << res.train_seconds << '\t' << res.eval_seconds << '\t' << (is_best ? 1 : 0) << '\t' << (stop.empty() ? "" : stop) << '\n';
      traj.flush();
      log << "epoch " << epoch + 1 << "/" << p.epochs << "  lr " << lr << "  loss " << train_loss << "  updates " << res.updates;
      if (evaluated) { log << "  val " << selectName(p.select) << " " << selected(vm, p.select) << (is_best ? " *" : ""); }
      log << "  train " << res.train_seconds << " s\n";
      if (!stop.empty()) { break; }
    }
    res.stop_reason = stop;
    if (res.updates == 0) { throw std::runtime_error("no optimizer step was taken -- training set empty after length grouping?"); }

    // restore the best checkpoint, evaluate, write back
    restore(model, best_state);
    res.param_l2_change = l2Change(model, stock);
    if (!(res.param_l2_change > 0)) { throw std::runtime_error("the best checkpoint equals the stock model -- nothing was learned (refusing to write an unchanged model)"); }
    auto t2 = Clock::now();
    predict(model, blocks, Unit::Val, p.batch_size, dev, pred);
    predict(model, blocks, Unit::Test, p.batch_size, dev, pred);
    res.tuned_val = metrics(d, pred, Unit::Val, ccs);
    res.tuned_test = metrics(d, pred, Unit::Test, ccs);
    res.eval_seconds += seconds(t2);
    model->to(torch::kCPU);
    storeWeights(onnx, model);
    onnx.write(p.model_out);
    res.model_out_sha256 = DIANNLibraryFile::hashFile(p.model_out);
    log << "tuned: val " << selectName(p.select) << " " << selected(res.tuned_val, p.select) << " " << unit
        << ", TEST sd " << res.tuned_test.sd << " calibrated " << res.tuned_test.calibrated_sd << " " << unit
        << " (stock " << res.stock_test.sd << " / " << res.stock_test.calibrated_sd << ")  best epoch " << res.best_epoch << "\n";

    nlohmann::json prov = {
      {"tool", "DIALibTune"}, {"schema_version", 1}, {"head", headName(p.head)}, {"units", ccs ? "1/K0 (model: CCS A^2)" : "minutes (model: rt_norm)"},
      {"libtorch", TORCH_VERSION}, {"device", p.device},
      {"recipe", {{"loss", "L1"}, {"optimizer", "Adam"}, {"lr", p.lr}, {"betas", {0.9, 0.999}}, {"eps", 1e-8}, {"weight_decay", 0.0}, {"clip_grad_norm", 1.0},
                  {"batch_size", p.batch_size}, {"epochs", p.epochs}, {"warmup", p.warmup}, {"schedule", "linear warmup then cosine, stepped per epoch"}, {"dropout", 0.1}, {"seed", p.seed}}},
      {"stopping", {{"eval_every", p.eval_every}, {"min_epochs", p.min_epochs}, {"patience", p.patience}, {"rel_tol", p.rel_tol}, {"abs_tol", p.abs_tol}, {"max_seconds", p.max_seconds}, {"select", selectName(p.select)}}},
      {"filter", {{"q_value", p.q_value}, {"min_charge", p.min_charge}, {"rt_spread_max", p.rt_spread_max}, {"rt_max_minutes", d.rt_max_minutes}}},
      {"inputs", {{"report", p.report}, {"run", d.run}, {"rows", d.rows}, {"model_in", p.model_in}, {"model_in_sha256", res.model_in_sha256}}},
      {"output", {{"model_out", p.model_out}, {"model_out_sha256", res.model_out_sha256}}},
      {"cohorts", {{"observations", res.observations}, {"units", res.units}, {"test", res.test}, {"val", res.val}, {"pool", res.pool}, {"training", res.training},
                   {"train_size", p.train_size}, {"train_frac", p.train_frac}, {"inner_val", p.inner_val}, {"rule", "test = crc32(pg)%5==0; val = crc32('val:'+pg)%7==0 of the rest; training = seeded shuffle prefix of the pool"},
                   {"rejected", d.rejected.json()}}},
      {"course", {{"epochs_run", res.epochs_run}, {"best_epoch", res.best_epoch}, {"updates", res.updates}, {"train_rows_per_epoch", train_rows},
                  {"train_seconds", res.train_seconds}, {"eval_seconds", res.eval_seconds}, {"stop_reason", res.stop_reason}, {"param_l2_change", res.param_l2_change}}},
      {"evaluation", {{"stock", {{"val", json(res.stock_val)}, {"test", json(res.stock_test)}}}, {"tuned", {{"val", json(res.tuned_val)}, {"test", json(res.tuned_test)}}}}}};
    std::ofstream(p.model_out + ".tune.json") << prov.dump(2) << "\n";
    return res;
  }
}
