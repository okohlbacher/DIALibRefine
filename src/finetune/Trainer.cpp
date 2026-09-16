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
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <unistd.h>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace ODIA::tune
{
  const char* headName(HeadKind h) { return h == HeadKind::CCS ? "ccs" : "rt"; }
  const char* selectName(Select s) { return s == Select::Rmse ? "rmse" : "calibrated_sd"; }

  namespace
  {
    using Clock = std::chrono::steady_clock;
    double seconds(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

    // ---- Parquet -------------------------------------------------------------

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

    /// Only the columns asked for are materialised: a DIA-NN report has ~90.
    std::shared_ptr<arrow::Table> readParquet(const std::string& path, const std::vector<std::string>& required,
                                              const std::vector<std::string>& optional)
    {
      auto infile = arrow::io::ReadableFile::Open(path);
      if (!infile.ok()) { throw std::runtime_error("cannot open " + path + ": " + infile.status().ToString()); }
      auto reader = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
      if (!reader.ok()) { throw std::runtime_error("not a Parquet file: " + path + ": " + reader.status().ToString()); }
      std::shared_ptr<arrow::Schema> schema;
      auto st = (*reader)->GetSchema(&schema);
      if (!st.ok()) { throw std::runtime_error("cannot read the schema of " + path); }
      std::vector<int> idx;
      for (const auto& n : required)
      {
        const int i = schema->GetFieldIndex(n);
        if (i < 0) { throw std::runtime_error("report lacks column " + n); }
        idx.push_back(i);
      }
      for (const auto& n : optional) { const int i = schema->GetFieldIndex(n); if (i >= 0) { idx.push_back(i); } }
      std::shared_ptr<arrow::Table> table;
      st = (*reader)->ReadTable(idx, &table);
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
      bool used = false;     ///< training, validation or test: the only units encoded
    };

    struct Rejections
    {
      std::size_t q = 0, charge = 0, decoy = 0, nonfinite = 0, no_protein_group = 0, unparsable = 0, unencodable = 0, spread = 0, pg_conflict = 0;
      nlohmann::json json() const
      {
        return {{"q_value", q}, {"charge", charge}, {"decoy", decoy}, {"nonfinite_or_out_of_range", nonfinite}, {"no_protein_group", no_protein_group},
                {"unparsable_sequence", unparsable}, {"unencodable_sequence", unencodable}, {"rt_spread", spread}, {"protein_group_conflict", pg_conflict}};
      }
    };

    struct Dataset
    {
      std::vector<Unit> units;
      std::size_t observations = 0, rows = 0;
      std::string run;
      double rt_max_minutes = 0;
      Rejections rejected;
    };

    /// The contract of tools/finetune.py's load_report(): finite RT >= 0, finite
    /// IM > 0 (CCS), finite m/z > 0, integral charge in 1..8, finite q, a
    /// non-empty protein group, no decoys, one run. Units whose observations
    /// disagree on the protein group are rejected, because the group decides
    /// the cohort and a first-row rule would let the row order decide it.
    Dataset loadReport(const TuneParams& p, std::ostream& log)
    {
      const bool ccs = p.head == HeadKind::CCS;
      std::vector<std::string> required{"Modified.Sequence", "Precursor.Charge", "Precursor.Mz", "RT", "Q.Value", "Protein.Group"};
      if (ccs) { required.push_back("IM"); }
      auto table = readParquet(p.report, required, {"Run", "Decoy"});
      Dataset d;
      d.rows = static_cast<std::size_t>(table->num_rows());
      const auto seq = toStrings(table->GetColumnByName("Modified.Sequence"));
      const auto pg = toStrings(table->GetColumnByName("Protein.Group"));
      const auto z = toDoubles(table->GetColumnByName("Precursor.Charge"));
      const auto mz = toDoubles(table->GetColumnByName("Precursor.Mz"));
      const auto rt = toDoubles(table->GetColumnByName("RT"));
      const auto q = toDoubles(table->GetColumnByName("Q.Value"));
      const auto im = ccs ? toDoubles(table->GetColumnByName("IM")) : std::vector<double>();
      auto run_col = table->GetColumnByName("Run");
      auto decoy_col = table->GetColumnByName("Decoy");
      const auto decoy = decoy_col ? toDoubles(decoy_col) : std::vector<double>();
      if (run_col)
      {
        const auto run = toStrings(run_col);
        std::set<std::string> runs(run.begin(), run.end());
        if (runs.size() != 1) { throw std::runtime_error("the report holds " + std::to_string(runs.size()) + " runs; fine-tuning is per run -- give a single-run report"); }
        d.run = *runs.begin();
      }

      // The charge floor is a CCS matter (z1 is censored at the ramp top); RT
      // units collapse charge states, so every charge contributes there.
      const int z_min = ccs ? p.min_charge : 1;
      struct Obs { double rt, im, mz; std::string pg; };
      std::map<std::pair<std::string, int>, std::vector<Obs>> by_key;
      for (std::size_t i = 0; i < d.rows; ++i)
      {
        if (!decoy.empty() && !(decoy[i] == 0)) { ++d.rejected.decoy; continue; }          // any truthy value, NaN included
        if (!std::isfinite(q[i]) || q[i] > p.q_value) { ++d.rejected.q; continue; }
        if (!std::isfinite(z[i]) || z[i] != std::floor(z[i]) || z[i] < 1 || z[i] > 8) { ++d.rejected.charge; continue; }
        const int zi = static_cast<int>(z[i]);
        if (zi < z_min) { ++d.rejected.charge; continue; }
        if (!std::isfinite(rt[i]) || rt[i] < 0 || !std::isfinite(mz[i]) || mz[i] <= 0) { ++d.rejected.nonfinite; continue; }
        if (ccs && (!std::isfinite(im[i]) || im[i] <= 0)) { ++d.rejected.nonfinite; continue; }
        if (pg[i].empty()) { ++d.rejected.no_protein_group; continue; }
        by_key[{seq[i], ccs ? zi : 0}].push_back({rt[i], ccs ? im[i] : NAN, mz[i], pg[i]});
        ++d.observations;
      }

      double rt_hi = 0;
      for (auto& [key, obs] : by_key)
      {
        Unit u;
        bool conflict = false;
        for (const auto& o : obs) { if (o.pg != obs.front().pg) { conflict = true; break; } }
        if (conflict) { ++d.rejected.pg_conflict; continue; }
        try { u.peptide = OpenMS::AASequence::fromString(key.first); }
        catch (const std::exception&) { ++d.rejected.unparsable; continue; }
        try { (void)PeptDeepEncoder::encode(u.peptide); }
        catch (const std::exception&) { ++d.rejected.unencodable; continue; }
        std::vector<double> vals;
        for (const auto& o : obs) { vals.push_back(ccs ? o.im : o.rt); }
        if (!ccs && obs.size() > 1)
        {
          const auto [lo, hi] = std::minmax_element(vals.begin(), vals.end());
          if (*hi - *lo > p.rt_spread_max) { ++d.rejected.spread; continue; }
        }
        u.charge = key.second;
        u.mz = obs.front().mz;            // one precursor, one m/z; the first as in the reference
        u.observed = median(vals);
        u.protein_group = obs.front().pg;
        if (!ccs) { rt_hi = std::max(rt_hi, u.observed); }
        d.units.push_back(std::move(u));
      }
      if (d.units.empty()) { throw std::runtime_error("no usable observations after filtering (q <= " + std::to_string(p.q_value) + ", charge >= " + std::to_string(z_min) + ")"); }

      if (!ccs)
      {
        if (p.rt_max_minutes > 0 && rt_hi > p.rt_max_minutes)
        { throw std::runtime_error("filter:rt_max_minutes = " + std::to_string(p.rt_max_minutes) + " but the report's RT reaches " + std::to_string(rt_hi) + " min; rt_norm would exceed 1"); }
        d.rt_max_minutes = p.rt_max_minutes > 0 ? p.rt_max_minutes : rt_hi;
        if (!(d.rt_max_minutes > 0)) { throw std::runtime_error("the RT scale is zero -- every observed RT is 0"); }
      }
      for (auto& u : d.units)
      {
        u.target = ccs ? ccsFromMobility(u.observed, u.mz, u.charge) : u.observed / d.rt_max_minutes;
        u.cohort = crc32(u.protein_group) % 5 == 0 ? Unit::Test
                 : crc32("val:" + u.protein_group) % 7 == 0 ? Unit::Val : Unit::Pool;
      }
      log << "report: " << d.rows << " rows, " << d.observations << " observations (q <= " << p.q_value << ", charge " << z_min << "..8), "
          << d.units.size() << " " << (ccs ? "sequence x charge" : "sequence") << " units (rejected: q " << d.rejected.q << ", charge " << d.rejected.charge
          << ", decoy " << d.rejected.decoy << ", non-finite/out of range " << d.rejected.nonfinite << ", no protein group " << d.rejected.no_protein_group
          << ", protein-group conflict " << d.rejected.pg_conflict << ", unparsable " << d.rejected.unparsable << ", unencodable " << d.rejected.unencodable
          << ", rt spread " << d.rejected.spread << ")\n";
      return d;
    }

    // ---- encoded tensors, one block per peptide length ------------------------

    struct Block
    {
      std::vector<std::size_t> unit;          ///< unit index per row
      torch::Tensor aa, mod_x, charges, target;
      std::vector<std::int64_t> train_rows, val_rows, test_rows;
    };

    /// Only the units marked `used` are encoded and uploaded: the pool a
    /// subsample did not draw stays on the CPU as strings.
    std::vector<Block> encode(const std::vector<Unit>& units, Unit::Cohort val_cohort, const torch::Device& dev)
    {
      std::vector<OpenMS::AASequence> peps; std::vector<std::size_t> which;
      for (std::size_t i = 0; i < units.size(); ++i) { if (units[i].used) { peps.push_back(units[i].peptide); which.push_back(i); } }
      std::vector<Block> blocks;
      for (const auto& group : PeptDeepEncoder::groupByLength(peps))
      {
        std::vector<OpenMS::AASequence> gp; std::vector<int> gz; std::vector<float> gy;
        Block blk;
        for (auto g : group)
        {
          const auto& u = units[which[g]];
          gp.push_back(u.peptide); gz.push_back(std::max(1, u.charge)); gy.push_back(static_cast<float>(u.target));
          blk.unit.push_back(which[g]);
        }
        auto b = PeptDeepEncoder::encode(gp, gz, 30.0f, "Lumos");
        const auto rows = static_cast<std::int64_t>(b.rows), L = static_cast<std::int64_t>(b.sequence_length);
        blk.aa = torch::from_blob(b.aa_indices.data(), {rows, L}, torch::kInt64).clone().to(dev);
        blk.mod_x = torch::from_blob(b.mod_x.data(), {rows, L, MOD_FEATURES}, torch::kFloat32).clone().to(dev);
        blk.charges = torch::from_blob(b.charges.data(), {rows, 1}, torch::kFloat32).clone().to(dev);
        blk.target = torch::from_blob(gy.data(), {rows}, torch::kFloat32).clone().to(dev);
        for (std::int64_t r = 0; r < rows; ++r)
        {
          const auto& u = units[blk.unit[static_cast<std::size_t>(r)]];
          if (u.cohort == Unit::Test) { blk.test_rows.push_back(r); }
          if (u.cohort == val_cohort) { blk.val_rows.push_back(r); }
          if (u.training) { blk.train_rows.push_back(r); }
        }
        blocks.push_back(std::move(blk));
      }
      return blocks;
    }

    torch::Tensor upload(const std::vector<std::int64_t>& rows, const torch::Device& dev)
    {
      return torch::from_blob(const_cast<std::int64_t*>(rows.data()), {static_cast<std::int64_t>(rows.size())}, torch::kInt64).clone().to(dev);
    }

    // ---- evaluation, deployed units ----------------------------------------------

    /// Predictions for one cohort of every block, written into pred[unit].
    void predict(Head& model, const std::vector<Block>& blocks, bool test, int batch, const torch::Device& dev, std::vector<double>& pred)
    {
      torch::NoGradGuard ng;
      model->eval();
      for (const auto& b : blocks)
      {
        const auto& rows = test ? b.test_rows : b.val_rows;
        if (rows.empty()) { continue; }
        auto all = upload(rows, dev);
        for (std::int64_t s = 0; s < all.size(0); s += batch)
        {
          auto idx = all.slice(0, s, std::min(all.size(0), s + batch));
          auto y = model->forward(b.aa.index_select(0, idx), b.mod_x.index_select(0, idx), b.charges.index_select(0, idx)).to(torch::kCPU).contiguous();
          const float* yp = y.data_ptr<float>();
          for (std::int64_t k = 0; k < y.size(0); ++k) { pred[b.unit[static_cast<std::size_t>(rows[static_cast<std::size_t>(s + k)])]] = yp[k]; }
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
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < d.units.size(); ++i)
      {
        const auto& u = d.units[i];
        if (u.cohort != which) { continue; }
        if (!std::isfinite(pred[i])) { ++nonfinite; continue; }
        const double p = ccs ? mobilityFromCCS(pred[i], u.mz, u.charge) : pred[i] * d.rt_max_minutes;
        obs.push_back(u.observed); dep.push_back(p);
        if (ccs) { by_z[u.charge].push_back(p - u.observed); }
      }
      Metrics m; m.n = obs.size(); m.nonfinite_predictions = nonfinite;
      // A NaN prediction is a broken model, not a missing row: it must not
      // leave the metric looking better by its absence.
      if (m.n < 3 || nonfinite > 0) { return m; }
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
      // least squares obs ~ a * pred + b; a constant prediction is a rank-
      // deficient fit that reduces to the mean (numpy.polyfit's answer, not NaN)
      const double n = static_cast<double>(m.n);
      const double mx = std::accumulate(dep.begin(), dep.end(), 0.0) / n, my = std::accumulate(obs.begin(), obs.end(), 0.0) / n;
      double sxx = 0, sxy = 0;
      for (std::size_t i = 0; i < m.n; ++i) { sxx += (dep[i] - mx) * (dep[i] - mx); sxy += (dep[i] - mx) * (obs[i] - my); }
      m.cal_slope = sxx > 0 ? sxy / sxx : 0.0;
      m.cal_intercept = my - m.cal_slope * mx;
      std::vector<double> cr(m.n);
      for (std::size_t i = 0; i < m.n; ++i) { cr[i] = obs[i] - (m.cal_slope * dep[i] + m.cal_intercept); }
      m.calibrated_sd = sd(cr);
      for (auto& [zc, v] : by_z)
      {
        if (v.size() < 30) { continue; }
        m.sd_by_charge[zc] = sd(v); m.n_by_charge[zc] = v.size();
        m.mean_by_charge[zc] = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
      }
      return m;
    }

    nlohmann::json json(const Metrics& m)
    {
      auto f = [](double x) { return std::isfinite(x) ? nlohmann::json(x) : nlohmann::json(nullptr); };
      nlohmann::json j = {{"n", m.n}, {"nonfinite_predictions", m.nonfinite_predictions}, {"rmse", f(m.rmse)}, {"sd", f(m.sd)}, {"mean_err", f(m.mean_err)}, {"p95", f(m.p95)},
                          {"calibrated_sd", f(m.calibrated_sd)}, {"cal_slope", f(m.cal_slope)}, {"cal_intercept", f(m.cal_intercept)}};
      for (auto& [z, s] : m.sd_by_charge) { j["by_charge"][std::to_string(z)] = {{"n", m.n_by_charge.at(z)}, {"sd", f(s)}, {"mean_err", f(m.mean_by_charge.at(z))}}; }
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

    /// torch.nn.utils.clip_grad_norm_ (max_norm, L2), computed entirely on the
    /// device: libtorch's own returns the norm as a host double, which is a
    /// synchronisation on every step.
    void clipGradNorm(const std::vector<torch::Tensor>& params, double max_norm)
    {
      std::vector<torch::Tensor> norms;
      for (const auto& p : params) { if (p.grad().defined()) { norms.push_back(p.grad().norm()); } }
      if (norms.empty()) { return; }
      auto total = torch::stack(norms).norm();
      auto coef = torch::clamp_max(max_norm / (total + 1e-6), 1.0);
      for (auto& p : params) { if (p.grad().defined()) { p.grad().mul_(coef); } }
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

    void writeAtomically(const std::string& path, const std::function<void(const std::string&)>& write)
    {
      // A unique scratch name: "<out>.part" could be the input, or a symlink to it.
      const std::string part = path + ".tmp-" + std::to_string(static_cast<long>(::getpid()));
      std::error_code ec0;
      if (std::filesystem::exists(part, ec0)) { throw std::runtime_error("scratch file already exists: " + part); }
      write(part);
      std::error_code ec;
      std::filesystem::rename(part, path, ec);
      if (ec) { throw std::runtime_error("cannot move " + part + " into place: " + ec.message()); }
    }
  }

  TuneResult finetune(const TuneParams& p, std::ostream& log)
  {
    const bool ccs = p.head == HeadKind::CCS;
    {
      std::error_code ec;
      const auto a = std::filesystem::weakly_canonical(p.model_in, ec), b = std::filesystem::weakly_canonical(p.model_out, ec);
      if (p.model_out == p.model_in || (!ec && a == b)) { throw std::runtime_error("-out must not be the stock model itself"); }
    }
    if (p.rel_tol < 0 || p.abs_tol < 0) { throw std::runtime_error("tolerances must be >= 0"); }
    if (p.train_frac < 0 || p.train_frac > 1) { throw std::runtime_error("train_frac must be in [0, 1]"); }
    if (ccs && p.min_charge < 2 && !p.allow_z1) { throw std::runtime_error("charge 1 is censored at the mobility ramp top on timsTOF; pass -filter:allow_z1 to train CCS on it anyway"); }
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
    if (const char* e = std::getenv("DLR_NO_MKLDNN"); e && *e && *e != '0') { at::globalContext().setUserEnabledMkldnn(false); log << "oneDNN disabled (DLR_NO_MKLDNN)\n"; }
    // pytorch.org's CUDA zips ship only cuDNN's loader shim; without the sub-
    // libraries on the library path cudnnCreate aborts the process (not an
    // exception). The native kernels are slower but always there.
    if (dev.is_cuda()) { at::globalContext().setUserEnabledCuDNN(p.cudnn); }
    torch::manual_seed(p.seed);
    log << "device " << p.device << (dev.is_cpu() ? " (" + std::to_string(torch::get_num_threads()) + " threads)" : (p.cudnn ? " (cuDNN on)" : " (cuDNN off)")) << ", libtorch " << TORCH_VERSION << "\n";

    // data + cohorts
    Dataset d = loadReport(p, log);
    // Without an inner validation cohort those units rejoin the pool and
    // selection runs on TEST -- which then stops being a held-out number.
    // The reference tool does the same; say it out loud.
    const Unit::Cohort val_cohort = p.inner_val ? Unit::Val : Unit::Test;
    if (!p.inner_val) { for (auto& u : d.units) { if (u.cohort == Unit::Val) { u.cohort = Unit::Pool; } } }
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
    res.val_is_test = !p.inner_val;
    if (!p.inner_val) { res.val = res.test; }
    // The reference tool's floor: below this the affine calibration behind the
    // selection metric fits noise, and a two-point cohort fits anything.
    if (res.val < 100 || res.test < 100) { throw std::runtime_error("validation or test cohort has fewer than 100 units (val " + std::to_string(res.val) + ", test " + std::to_string(res.test) + "); the report is too small to fine-tune on with held-out selection"); }
    std::mt19937 rng(p.seed);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::size_t n_train = pool.size();
    if (p.train_size) { n_train = std::min(n_train, p.train_size); }
    else if (p.train_frac > 0) { n_train = std::min(n_train, std::max<std::size_t>(1, static_cast<std::size_t>(std::nearbyint(p.train_frac * static_cast<double>(pool.size()))))); }
    if (n_train == 0) { throw std::runtime_error("training set is empty (pool " + std::to_string(pool.size()) + ")"); }
    for (std::size_t k = 0; k < n_train; ++k) { d.units[pool[k]].training = true; }
    res.training = n_train;
    for (auto& u : d.units) { u.used = u.training || u.cohort == Unit::Test || u.cohort == val_cohort; }
    log << "cohorts: test " << res.test << " (protein-held-out), val " << res.val << (p.inner_val ? " (protein-held-out)" : " (= TEST: selection is optimistic, TEST is no longer held out)")
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
    auto blocks = encode(d.units, val_cohort, dev);
    std::size_t train_rows = 0, encoded = 0; for (const auto& b : blocks) { train_rows += b.train_rows.size(); encoded += b.unit.size(); }
    log << "encoded " << encoded << " units in " << blocks.size() << " length groups, " << seconds(t_enc) << " s\n";

    std::vector<double> pred(d.units.size(), NAN);
    auto t_eval = Clock::now();
    predict(model, blocks, false, p.batch_size, dev, pred);
    predict(model, blocks, true, p.batch_size, dev, pred);
    res.stock_val = metrics(d, pred, val_cohort, ccs);
    res.stock_test = metrics(d, pred, Unit::Test, ccs);
    res.eval_seconds += seconds(t_eval);
    const char* unit = ccs ? "1/K0" : "min";
    const double stock_metric = selected(res.stock_val, p.select);
    if (!std::isfinite(stock_metric)) { throw std::runtime_error("the stock model's validation metric is not finite -- cannot select against it"); }
    log << "stock: val " << selectName(p.select) << " " << stock_metric << " " << unit
        << ", TEST sd " << res.stock_test.sd << " calibrated " << res.stock_test.calibrated_sd << " " << unit << "\n";

    // optimizer: Adam on the trainable parameters only (h0/c0 stay frozen)
    auto params = model->trainable();
    torch::optim::Adam opt(params, torch::optim::AdamOptions(p.lr).betas({0.9, 0.999}).eps(1e-8));
    auto setLr = [&](double lr) { for (auto& g : opt.param_groups()) { static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr); } };

    std::ofstream traj(p.model_out + ".trajectory.tsv");
    if (!traj) { throw std::runtime_error("cannot write " + p.model_out + ".trajectory.tsv"); }
    traj << "epoch\tlr\tupdates\ttrain_loss\tval_rmse\tval_sd\tval_mean_err\tval_p95\tval_calibrated_sd\ttrain_s\teval_s\tbest\tstop\n";

    // Selection starts from the STOCK model: a checkpoint is exported only if
    // it beat stock on validation. The anchor-patience rule is the reference
    // tool's: the anchor is the first trained checkpoint (NOT stock -- a run
    // that degrades first and recovers must not be stopped for it), progress
    // = beating the anchor by max(abs_tol, rel_tol * anchor); patience counts
    // EPOCHS since the last progress; never inside warmup.
    double best = stock_metric, anchor = NAN;   // anchor: the first trained checkpoint, as the reference does
    int anchor_epoch = 0;
    std::vector<torch::Tensor> best_state = stock;
    res.best_epoch = 0;
    std::string stop;
    std::vector<std::size_t> block_order(blocks.size());
    std::iota(block_order.begin(), block_order.end(), 0);
    const int no_stop_before = std::max(p.min_epochs, p.warmup);

    for (int epoch = 0; epoch < p.epochs; ++epoch)
    {
      const double lr = p.lr * lrLambda(epoch, p.warmup, p.epochs);
      setLr(lr);
      model->train();
      auto t0 = Clock::now();
      auto loss_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
      std::size_t loss_n = 0;
      std::shuffle(block_order.begin(), block_order.end(), rng);
      for (auto bi : block_order)
      {
        auto& b = blocks[bi];
        if (b.train_rows.empty()) { continue; }
        std::shuffle(b.train_rows.begin(), b.train_rows.end(), rng);
        auto perm = upload(b.train_rows, dev);                     // one upload per block and epoch
        for (std::int64_t s = 0; s < perm.size(0); s += p.batch_size)
        {
          auto idx = perm.slice(0, s, std::min(perm.size(0), s + p.batch_size));
          opt.zero_grad();
          auto y = model->forward(b.aa.index_select(0, idx), b.mod_x.index_select(0, idx), b.charges.index_select(0, idx));
          auto loss = torch::l1_loss(y, b.target.index_select(0, idx));
          loss.backward();
          clipGradNorm(params, 1.0);
          opt.step();
          ++res.updates;
          loss_sum += loss.detach().to(torch::kFloat64) * static_cast<double>(idx.size(0));   // no host sync per step
          loss_n += static_cast<std::size_t>(idx.size(0));
        }
      }
      if (dev.is_cuda()) { torch::cuda::synchronize(); }   // the timer must include the queued kernels
      res.train_seconds += seconds(t0);
      res.epochs_run = epoch + 1;
      const double train_loss = loss_n ? loss_sum.item<double>() / static_cast<double>(loss_n) : NAN;
      if (loss_n && !std::isfinite(train_loss)) { throw std::runtime_error("non-finite loss at epoch " + std::to_string(epoch + 1) + " -- training diverged"); }

      // A budget or horizon stop evaluates the state it stops at, so no run
      // ends without a checkpoint having been looked at.
      const bool budget_hit = p.max_seconds > 0 && res.train_seconds >= p.max_seconds;
      const bool last = epoch + 1 == p.epochs;
      const bool evaluate = (epoch + 1) % p.eval_every == 0 || last || budget_hit;
      Metrics vm; bool is_best = false;
      if (evaluate)
      {
        auto t1 = Clock::now();
        predict(model, blocks, false, p.batch_size, dev, pred);
        vm = metrics(d, pred, val_cohort, ccs);
        res.eval_seconds += seconds(t1);
        const double m = selected(vm, p.select);
        if (!std::isfinite(m)) { throw std::runtime_error("non-finite validation metric at epoch " + std::to_string(epoch + 1)); }
        if (m < best) { best = m; res.best_epoch = epoch + 1; best_state = snapshot(model); is_best = true; }
        if (!std::isfinite(anchor) || anchor - m >= std::max(p.abs_tol, p.rel_tol * anchor)) { anchor = m; anchor_epoch = epoch + 1; }
        if (epoch + 1 >= no_stop_before && epoch + 1 - anchor_epoch >= p.patience)
        { stop = "patience (" + std::to_string(epoch + 1 - anchor_epoch) + " epochs without progress of max(" + std::to_string(p.abs_tol) + ", " + std::to_string(100 * p.rel_tol) + "% of the anchor))"; }
      }
      if (stop.empty() && budget_hit) { stop = "max_seconds"; }
      if (stop.empty() && last) { stop = "horizon"; }

      traj << epoch + 1 << '\t' << lr << '\t' << res.updates << '\t' << train_loss << '\t';
      if (evaluate) { traj << vm.rmse << '\t' << vm.sd << '\t' << vm.mean_err << '\t' << vm.p95 << '\t' << vm.calibrated_sd; }
      else { traj << "\t\t\t\t"; }
      traj << '\t' << res.train_seconds << '\t' << res.eval_seconds << '\t' << (is_best ? 1 : 0) << '\t' << stop << '\n';
      traj.flush();
      log << "epoch " << epoch + 1 << "/" << p.epochs << "  lr " << lr << "  loss " << train_loss << "  updates " << res.updates;
      if (evaluate) { log << "  val " << selectName(p.select) << " " << selected(vm, p.select) << (is_best ? " *" : ""); }
      log << "  train " << res.train_seconds << " s\n";
      if (!stop.empty()) { break; }
    }
    res.stop_reason = stop;
    traj.close();
    if (!traj) { throw std::runtime_error("cannot write " + p.model_out + ".trajectory.tsv"); }
    if (res.updates == 0) { throw std::runtime_error("no optimizer step was taken -- training set empty after length grouping?"); }

    // restore the best checkpoint, evaluate, write back
    restore(model, best_state);
    res.param_l2_change = l2Change(model, stock);
    auto t2 = Clock::now();
    predict(model, blocks, false, p.batch_size, dev, pred);
    predict(model, blocks, true, p.batch_size, dev, pred);
    res.tuned_val = metrics(d, pred, val_cohort, ccs);
    res.tuned_test = metrics(d, pred, Unit::Test, ccs);
    res.eval_seconds += seconds(t2);
    const bool improved = res.best_epoch > 0 && res.param_l2_change > 0;

    nlohmann::json prov = {
      {"tool", "DIALibTune"}, {"schema_version", 1}, {"head", headName(p.head)}, {"units", ccs ? "1/K0 (model: CCS A^2)" : "minutes (model: rt_norm)"},
      {"libtorch", TORCH_VERSION}, {"device", p.device}, {"cudnn", p.cudnn},
      {"recipe", {{"loss", "L1"}, {"optimizer", "Adam"}, {"lr", p.lr}, {"betas", {0.9, 0.999}}, {"eps", 1e-8}, {"weight_decay", 0.0}, {"clip_grad_norm", 1.0},
                  {"batch_size", p.batch_size}, {"epochs", p.epochs}, {"warmup", p.warmup}, {"schedule", "linear warmup then cosine, stepped per epoch"}, {"dropout", 0.1}, {"seed", p.seed}}},
      {"stopping", {{"eval_every", p.eval_every}, {"min_epochs", p.min_epochs}, {"patience_epochs", p.patience}, {"rel_tol", p.rel_tol}, {"abs_tol", p.abs_tol}, {"max_seconds", p.max_seconds},
                    {"select", selectName(p.select)}, {"rule", "progress = beat the anchor by max(abs_tol, rel_tol*anchor); stop after `patience` epochs without progress, never before max(min_epochs, warmup); selection starts from the stock model"}}},
      {"filter", {{"q_value", p.q_value}, {"min_charge", ccs ? p.min_charge : 1}, {"allow_z1", p.allow_z1}, {"rt_spread_max", p.rt_spread_max}, {"rt_max_minutes", d.rt_max_minutes}}},
      {"inputs", {{"report", p.report}, {"run", d.run}, {"rows", d.rows}, {"model_in", p.model_in}, {"model_in_sha256", res.model_in_sha256}}},
      {"cohorts", {{"observations", res.observations}, {"units", res.units}, {"test", res.test}, {"val", res.val}, {"val_is_test", res.val_is_test}, {"pool", res.pool}, {"training", res.training},
                   {"train_size", p.train_size}, {"train_frac", p.train_frac}, {"rule", "test = crc32(pg)%5==0; val = crc32('val:'+pg)%7==0 of the rest; training = seeded shuffle prefix of the pool"},
                   {"rejected", d.rejected.json()}}},
      {"course", {{"epochs_run", res.epochs_run}, {"best_epoch", res.best_epoch}, {"updates", res.updates}, {"train_rows_per_epoch", train_rows}, {"encoded_units", encoded},
                  {"train_seconds", res.train_seconds}, {"eval_seconds", res.eval_seconds}, {"stop_reason", res.stop_reason}, {"param_l2_change", res.param_l2_change}, {"exported", improved}}},
      {"evaluation", {{"stock", {{"val", json(res.stock_val)}, {"test", json(res.stock_test)}}}, {"tuned", {{"val", json(res.tuned_val)}, {"test", json(res.tuned_test)}}}}}};
    writeAtomically(p.model_out + ".tune.json", [&](const std::string& part)
    {
      std::ofstream o(part); o << prov.dump(2) << "\n"; o.close();
      if (!o) { throw std::runtime_error("cannot write " + part); }
    });

    if (!improved)
    {
      throw std::runtime_error("no checkpoint beat the stock model on validation (" + std::string(selectName(p.select)) + " " + std::to_string(stock_metric) + " " + unit
                               + ", best trained " + std::to_string(best) + "); nothing written to " + p.model_out + " -- see " + p.model_out + ".tune.json");
    }
    model->to(torch::kCPU);
    storeWeights(onnx, model);
    writeAtomically(p.model_out, [&](const std::string& part) { onnx.write(part); });
    res.model_out_sha256 = DIANNLibraryFile::hashFile(p.model_out);
    prov["output"] = {{"model_out", p.model_out}, {"model_out_sha256", res.model_out_sha256}};
    writeAtomically(p.model_out + ".tune.json", [&](const std::string& part)
    {
      std::ofstream o(part); o << prov.dump(2) << "\n"; o.close();
      if (!o) { throw std::runtime_error("cannot write " + part); }
    });
    log << "tuned: val " << selectName(p.select) << " " << selected(res.tuned_val, p.select) << " " << unit
        << ", TEST sd " << res.tuned_test.sd << " calibrated " << res.tuned_test.calibrated_sd << " " << unit
        << " (stock " << res.stock_test.sd << " / " << res.stock_test.calibrated_sd << ")  best epoch " << res.best_epoch << "\n";
    return res;
  }
}
