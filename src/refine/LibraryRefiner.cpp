// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <odia/LibraryRefiner.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ODIA
{
  namespace
  {
    /// UniMod accession <-> interchange name. Canonical form is the accession:
    /// it is unambiguous, whereas the names differ between tools and some are
    /// synonyms of each other.
    struct ModAlias { const char* name; const char* unimod; };
    constexpr ModAlias MOD_ALIASES[] = {
      {"Acetyl", "UniMod:1"},
      {"Carbamidomethyl", "UniMod:4"},
      {"Carbamyl", "UniMod:5"},
      {"Deamidated", "UniMod:7"},
      {"Phospho", "UniMod:21"},
      {"Pyro-carbamidomethyl", "UniMod:26"},
      {"Glu->pyro-Glu", "UniMod:27"},
      {"Gln->pyro-Glu", "UniMod:28"},
      {"Oxidation", "UniMod:35"},
      {"Methyl", "UniMod:34"},
      {"Dimethyl", "UniMod:36"},
      {"Trimethyl", "UniMod:37"},
      {"GG", "UniMod:121"},
      {"Label:13C(6)15N(2)", "UniMod:259"},
      {"Label:13C(6)15N(4)", "UniMod:267"},
    };

    bool isNaN(float v) { return std::isnan(v); }

    double sd(const std::vector<double>& v, double mean)
    {
      if (v.size() < 2) { return 0.0; }
      double acc = 0.0;
      for (double x : v) { const double d = x - mean; acc += d * d; }
      return std::sqrt(acc / static_cast<double>(v.size() - 1));
    }

    double quantile(std::vector<double> v, double q)
    {
      if (v.empty()) { return 0.0; }
      const std::size_t i = static_cast<std::size_t>(q * (static_cast<double>(v.size()) - 1));
      std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
      return v[i];
    }

    /// Column lookup that tolerates the aliases the two dialects use.
    std::shared_ptr<arrow::ChunkedArray> column(const std::shared_ptr<arrow::Table>& t,
                                                std::initializer_list<const char*> names)
    {
      for (const char* n : names)
      {
        const int i = t->schema()->GetFieldIndex(n);
        if (i >= 0) { return t->column(i); }
      }
      return nullptr;
    }

    /// Materialise a whole column once.
    ///
    /// ChunkedArray::GetScalar allocates a Scalar per cell and searches the
    /// chunk list on every call; at 349k rows x 8 columns that dominated the
    /// run. Casting the column once and reading the buffer is the same answer,
    /// two orders of magnitude cheaper.
    std::vector<double> toDoubles(const std::shared_ptr<arrow::ChunkedArray>& c,
                                  std::int64_t rows, double fallback)
    {
      std::vector<double> out(static_cast<std::size_t>(rows), fallback);
      if (!c) { return out; }
      auto casted = arrow::compute::Cast(c, arrow::float64());
      if (!casted.ok()) { return out; }
      const auto chunked = casted->chunked_array();
      std::size_t at = 0;
      for (const auto& chunk : chunked->chunks())
      {
        const auto& d = static_cast<const arrow::DoubleArray&>(*chunk);
        for (std::int64_t i = 0; i < d.length(); ++i, ++at)
        { if (d.IsValid(i)) { out[at] = d.Value(i); } }
      }
      return out;
    }

    std::vector<std::string> toStrings(const std::shared_ptr<arrow::ChunkedArray>& c,
                                       std::int64_t rows)
    {
      std::vector<std::string> out(static_cast<std::size_t>(rows));
      if (!c) { return out; }
      std::size_t at = 0;
      for (const auto& chunk : c->chunks())
      {
        if (chunk->type_id() == arrow::Type::STRING)
        {
          const auto& a = static_cast<const arrow::StringArray&>(*chunk);
          for (std::int64_t i = 0; i < a.length(); ++i, ++at)
          { if (a.IsValid(i)) { out[at] = a.GetString(i); } }
        }
        else if (chunk->type_id() == arrow::Type::LARGE_STRING)
        {
          const auto& a = static_cast<const arrow::LargeStringArray&>(*chunk);
          for (std::int64_t i = 0; i < a.length(); ++i, ++at)
          { if (a.IsValid(i)) { out[at] = a.GetString(i); } }
        }
        else { at += static_cast<std::size_t>(chunk->length()); }
      }
      return out;
    }
  }

  std::string canonicalModifiedSequence(std::string_view seq)
  {
    std::string out;
    out.reserve(seq.size());
    for (std::size_t i = 0; i < seq.size();)
    {
      const char c = seq[i];
      if (c != '(' && c != '[') { out.push_back(c); ++i; continue; }

      const char close = (c == '(') ? ')' : ']';
      const std::size_t end = seq.find(close, i + 1);
      if (end == std::string_view::npos) { out.append(seq.substr(i)); break; }

      std::string_view body = seq.substr(i + 1, end - i - 1);
      // A leading '+'/'-' mass shift has no accession; leave it verbatim so it
      // never silently collides with a named modification.
      std::string canonical(body);
      for (const auto& a : MOD_ALIASES)
      {
        if (body == a.name) { canonical = a.unimod; break; }
        // DIA-NN also writes the bare accession without the "UniMod:" prefix in
        // some exports.
        if (body.size() > 7 && body.substr(0, 7) == "UniMod:") { canonical = std::string(body); break; }
      }
      out.push_back('(');
      out.append(canonical);
      out.push_back(')');
      i = end + 1;
    }
    return out;
  }

  std::string LibraryRefiner::key(std::string_view modified_sequence, int charge)
  {
    return canonicalModifiedSequence(modified_sequence) + "/" + std::to_string(charge);
  }

  LibraryRefiner::ObsMap LibraryRefiner::readObservations(const std::string& path,
                                                          const RefineParams& p,
                                                          RefineStats& stats)
  {
    std::shared_ptr<arrow::Table> table;
    {
      auto infile = arrow::io::ReadableFile::Open(path);
      if (!infile.ok()) { throw std::runtime_error("cannot open reference: " + path); }
      auto reader_result = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool());
      if (!reader_result.ok()) { throw std::runtime_error("not a readable Parquet file: " + path); }
      std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);
      const auto st = reader->ReadTable(&table);
      if (!st.ok()) { throw std::runtime_error("cannot read reference table: " + path); }
    }

    auto seq_c    = column(table, {"Modified.Sequence", "ModifiedPeptideSequence", "FullUniModPeptideName"});
    auto charge_c = column(table, {"Precursor.Charge", "PrecursorCharge"});
    auto rt_c     = column(table, {"RT", "NormalizedRetentionTime", "Tr_recalibrated"});
    auto im_c     = column(table, {"IM", "PrecursorIonMobility", "IonMobility"});
    auto q_c      = column(table, {"Q.Value", "QValue"});
    auto gq_c     = column(table, {"Global.Q.Value", "Global.Peptidoform.Q.Value"});
    auto pgq_c    = column(table, {"PG.Q.Value", "Protein.Q.Value", "Global.PG.Q.Value"});
    auto decoy_c  = column(table, {"Decoy"});
    auto qty_c    = column(table, {"Precursor.Quantity", "Precursor.Normalised", "Ms1.Area",
                                   "Relative.Intensity", "LibraryIntensity"});

    if (!seq_c || !charge_c)
    {
      throw std::runtime_error("reference has no Modified.Sequence / Precursor.Charge column; "
                               "it is not a DIA-NN report or library");
    }

    const std::int64_t rows = table->num_rows();
    stats.ids_rows = static_cast<std::size_t>(rows);

    const auto seq    = toStrings(seq_c, rows);
    const auto charge = toDoubles(charge_c, rows, 0.0);
    const auto rt     = toDoubles(rt_c, rows, std::numeric_limits<double>::quiet_NaN());
    const auto im     = toDoubles(im_c, rows, std::numeric_limits<double>::quiet_NaN());
    const auto qv     = toDoubles(q_c, rows, 0.0);
    const auto gq     = toDoubles(gq_c, rows, 0.0);
    const auto pgq    = toDoubles(pgq_c, rows, 0.0);
    const auto decoy  = toDoubles(decoy_c, rows, 0.0);
    const auto qty    = toDoubles(qty_c, rows, 0.0);

    ObsMap out;
    out.reserve(static_cast<std::size_t>(rows) / 4 + 16);
    std::unordered_map<std::string, std::uint32_t> row_counts;

    for (std::int64_t r = 0; r < rows; ++r)
    {
      const std::size_t i = static_cast<std::size_t>(r);
      if (decoy_c && decoy[i] != 0.0) { continue; }

      const std::string k = key(seq[i], static_cast<int>(charge[i]));
      ++row_counts[k];

      // Gate here rather than later: a row failing any of the three q-value
      // levels is not a weaker observation, it is not an observation.
      if (q_c && qv[i] > p.q_precursor) { continue; }
      if (gq_c && gq[i] > p.q_global) { continue; }
      if (pgq_c && pgq[i] > p.q_protein) { continue; }

      Observation o;
      o.rt = static_cast<float>(rt[i]);
      o.im = static_cast<float>(im[i]);
      o.q = static_cast<float>(qv[i]);
      o.quality = static_cast<float>(qty[i]);

      auto it = out.find(k);
      if (it == out.end()) { out.emplace(k, o); continue; }

      const bool better = (p.dedup == RefineParams::Dedup::BestQuality)
                            ? (o.quality > it->second.quality)
                            : (o.q < it->second.q);
      if (better) { it->second = o; }
    }

    for (auto& [k, o] : out) { o.fragments = row_counts[k]; }

    if (p.min_fragments > 0)
    {
      for (auto it = out.begin(); it != out.end();)
      {
        if (it->second.fragments < p.min_fragments) { it = out.erase(it); }
        else { ++it; }
      }
    }

    if (p.im_ramp_guard > 0.0)
    {
      double hi = -std::numeric_limits<double>::infinity();
      for (const auto& [k, o] : out) { if (!isNaN(o.im)) { hi = std::max(hi, static_cast<double>(o.im)); } }
      if (std::isfinite(hi))
      {
        for (auto it = out.begin(); it != out.end();)
        {
          if (!isNaN(it->second.im) && it->second.im > hi - p.im_ramp_guard)
          { ++stats.ids_ramp_dropped; it = out.erase(it); }
          else { ++it; }
        }
      }
    }

    stats.ids_precursors = row_counts.size();
    stats.ids_passing = out.size();
    return out;
  }

  void LibraryRefiner::refine(Library& library, const ObsMap& obs,
                              const RefineParams& p, RefineStats& stats)
  {
    if (p.write_intensity)
    {
      throw std::runtime_error(
        "-write_intensity is not implemented. The paper that motivates this tool measures "
        "fragment-intensity replacement as a wash -- 'RMSD in relative fragment ion intensity "
        "remain comparable between approaches' across three separate figures -- so it is the one "
        "component with no evidence behind it. It is refused rather than stubbed so that no arm "
        "can silently run without it.");
    }

    auto& pre = library.precursors();
    const std::size_t n = library.precursorCount();
    stats.library_before = n;

    std::vector<std::size_t> keep;
    keep.reserve(n);
    std::vector<double> rt_resid, im_resid;
    std::vector<std::size_t> matched_idx;
    matched_idx.reserve(obs.size());

    std::unordered_map<std::string, char> used;
    used.reserve(obs.size());

    for (std::size_t i = 0; i < n; ++i)
    {
      const std::string_view seq = library.strings().get(pre.modified_sequence[i]);
      const std::string k = key(seq, static_cast<int>(pre.charge[i]));

      auto it = obs.find(k);
      const bool is_decoy = pre.decoy[i] != 0;

      // A decoy is kept exactly when its target is, and for the same reason it
      // gets the same coordinates: ODIA's decoys carry the TARGET's sequence
      // with shifted fragments (odia.decoy_semantics), so they share this key.
      //
      // Keeping unmatched decoys "to preserve the pairing" is the wrong fix and
      // was the first thing this tool got wrong: it produced a library of 36,574
      // matched targets and 4,625,804 orphan decoys -- 99.2% decoy, which is not
      // a conservative FDR but a meaningless one.
      if (it == obs.end())
      {
        if (!p.filter) { keep.push_back(i); }
        continue;
      }

      keep.push_back(i);
      matched_idx.push_back(i);

      // Count and measure TARGETS only. A decoy sharing its target's key would
      // otherwise double every statistic -- the first run reported "matched
      // 72,979 of 36,574 reference precursors (199.5%)".
      if (is_decoy) { ++stats.decoys_kept; continue; }

      used[k] = 1;
      ++stats.matched;

      if (p.write_rt && !isNaN(it->second.rt) && !isNaN(pre.irt[i]))
      { rt_resid.push_back(static_cast<double>(pre.irt[i]) - it->second.rt); }
      if (p.write_im && !isNaN(it->second.im) && !isNaN(pre.im[i]))
      { im_resid.push_back(static_cast<double>(pre.im[i]) - it->second.im); }
    }

    stats.ids_unmatched = obs.size() - used.size();
    stats.join_looks_broken = (!obs.empty() && stats.matched * 20 < obs.size());

    // Residuals are computed BEFORE the write, because after it they are zero.
    auto summarise = [](std::vector<double>& v, double& mean, double& s, double& p95, std::size_t& cnt)
    {
      cnt = v.size();
      if (v.empty()) { return; }
      mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
      s = sd(v, mean);
      std::vector<double> abs_v;
      abs_v.reserve(v.size());
      for (double x : v) { abs_v.push_back(std::abs(x)); }
      p95 = quantile(std::move(abs_v), 0.95);
    };
    summarise(rt_resid, stats.rt_resid_mean, stats.rt_resid_sd, stats.rt_resid_p95, stats.rt_resid_n);
    summarise(im_resid, stats.im_resid_mean, stats.im_resid_sd, stats.im_resid_p95, stats.im_resid_n);

    // Write. Min-max rescaling needs the observed range over the matched set,
    // so it is a second pass rather than a per-row transform.
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    if (p.rt_unit == RefineParams::RtUnit::MinMax)
    {
      for (std::size_t i : matched_idx)
      {
        const std::string k = key(library.strings().get(pre.modified_sequence[i]),
                                  static_cast<int>(pre.charge[i]));
        const float rt = obs.at(k).rt;
        if (!isNaN(rt)) { lo = std::min(lo, static_cast<double>(rt)); hi = std::max(hi, static_cast<double>(rt)); }
      }
    }

    for (std::size_t i : matched_idx)
    {
      const std::string k = key(library.strings().get(pre.modified_sequence[i]),
                                static_cast<int>(pre.charge[i]));
      const Observation& o = obs.at(k);

      if (p.write_rt && !isNaN(o.rt))
      {
        double v = o.rt;
        if (p.rt_unit == RefineParams::RtUnit::MinMax && hi > lo)
        { v = 100.0 * (v - lo) / (hi - lo); }
        pre.irt[i] = static_cast<float>(v);
        ++stats.rt_written;
      }
      if (p.write_im && !isNaN(o.im))
      {
        pre.im[i] = o.im;
        // CCS was derived from the PREDICTED 1/K0 and is now inconsistent with
        // it. Leaving a stale value is worse than leaving none: a consumer that
        // prefers CCS would silently use the prediction it was told was replaced.
        if (i < pre.ccs.size()) { pre.ccs[i] = std::numeric_limits<float>::quiet_NaN(); }
        ++stats.im_written;
      }
    }

    if (p.filter && keep.size() != n)
    {
      library = library.subsetByIndex(keep);
    }
    stats.library_after = library.precursorCount();
  }
}
