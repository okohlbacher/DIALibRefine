// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <odia/LibraryRefiner.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ODIA
{
  namespace
  {
    /// UniMod accession <-> interchange name. Canonical form is the accession:
    /// it is unambiguous, whereas the names differ between tools.
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
      {"Methyl", "UniMod:34"},
      {"Oxidation", "UniMod:35"},
      {"Dimethyl", "UniMod:36"},
      {"Trimethyl", "UniMod:37"},
      {"GG", "UniMod:121"},
      {"Label:13C(6)15N(2)", "UniMod:259"},
      {"Label:13C(6)15N(4)", "UniMod:267"},
    };

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    bool finite(double v) { return std::isfinite(v); }

    double mean(const std::vector<double>& v)
    { return v.empty() ? NaN : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

    double sd(const std::vector<double>& v, double m)
    {
      if (v.size() < 2) { return NaN; }
      double acc = 0.0;
      for (double x : v) { const double d = x - m; acc += d * d; }
      return std::sqrt(acc / static_cast<double>(v.size() - 1));
    }

    /// Lower nearest-rank quantile: element at floor(q*(n-1)) of the sorted values.
    double quantile(std::vector<double> v, double q)
    {
      if (v.empty()) { return NaN; }
      const std::size_t i = static_cast<std::size_t>(q * (static_cast<double>(v.size()) - 1));
      std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
      return v[i];
    }

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

    /// Materialise a whole column once. Nulls and cast failures become NaN --
    /// and NaN is REJECTED by every consumer below, never treated as 0.
    std::vector<double> toDoubles(const std::shared_ptr<arrow::ChunkedArray>& c, std::int64_t rows)
    {
      std::vector<double> out(static_cast<std::size_t>(rows), NaN);
      if (!c) { return out; }
      auto casted = arrow::compute::Cast(c, arrow::float64());
      if (!casted.ok()) { throw std::runtime_error("reference column cannot be read as numbers: " + casted.status().ToString()); }
      std::size_t at = 0;
      for (const auto& chunk : casted->chunked_array()->chunks())
      {
        const auto& d = static_cast<const arrow::DoubleArray&>(*chunk);
        for (std::int64_t i = 0; i < d.length(); ++i, ++at)
        { if (d.IsValid(i)) { out[at] = d.Value(i); } }
      }
      return out;
    }

    std::vector<std::string> toStrings(const std::shared_ptr<arrow::ChunkedArray>& c, std::int64_t rows)
    {
      std::vector<std::string> out(static_cast<std::size_t>(rows));
      if (!c) { return out; }
      std::size_t at = 0;
      for (const auto& chunk : c->chunks())
      {
        if (chunk->type_id() == arrow::Type::STRING)
        {
          const auto& a = static_cast<const arrow::StringArray&>(*chunk);
          for (std::int64_t i = 0; i < a.length(); ++i, ++at) { if (a.IsValid(i)) { out[at] = a.GetString(i); } }
        }
        else if (chunk->type_id() == arrow::Type::LARGE_STRING)
        {
          const auto& a = static_cast<const arrow::LargeStringArray&>(*chunk);
          for (std::int64_t i = 0; i < a.length(); ++i, ++at) { if (a.IsValid(i)) { out[at] = a.GetString(i); } }
        }
        else { throw std::runtime_error("reference text column has an unsupported type: " + chunk->type()->ToString()); }
      }
      return out;
    }

    /// Is this modified sequence's token a name we know or an accession?
    bool knownModToken(std::string_view body)
    {
      if (body.size() > 7 && body.substr(0, 7) == "UniMod:") { return true; }
      for (const auto& a : MOD_ALIASES) { if (body == a.name) { return true; } }
      return false;
    }
  }

  std::string canonicalModifiedSequence(std::string_view seq, std::size_t* unknown)
  {
    std::string out;
    out.reserve(seq.size());
    for (std::size_t i = 0; i < seq.size();)
    {
      const char c = seq[i];
      if (c != '(' && c != '[') { out.push_back(c); ++i; continue; }

      // Balanced scan: nested brackets belong to the token. The previous
      // implementation stopped at the FIRST closing bracket, so
      // K(Label:13C(6)15N(2)) became K(Label:13C(6) and never matched anything.
      const char open = c, close = (c == '(') ? ')' : ']';
      std::size_t depth = 0, end = std::string_view::npos;
      for (std::size_t j = i; j < seq.size(); ++j)
      {
        if (seq[j] == open) { ++depth; }
        else if (seq[j] == close && --depth == 0) { end = j; break; }
      }
      if (end == std::string_view::npos) { out.append(seq.substr(i)); break; }

      const std::string_view body = seq.substr(i + 1, end - i - 1);
      std::string canonical(body);
      bool known = false;
      for (const auto& a : MOD_ALIASES)
      { if (body == a.name) { canonical = a.unimod; known = true; break; } }
      if (!known && body.size() > 7 && body.substr(0, 7) == "UniMod:") { known = true; }
      // Anything else -- a bare mass shift like (+57.0215), a bare number, an
      // unknown name -- passes through verbatim so it can never silently
      // collide with a known modification, and is counted.
      if (!known && unknown) { ++*unknown; }
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
    auto run_c    = column(table, {"Run"});
    auto rt_c     = column(table, {"RT", "NormalizedRetentionTime", "Tr_recalibrated"});
    auto im_c     = column(table, {"IM", "PrecursorIonMobility", "IonMobility"});
    auto q_c      = column(table, {"Q.Value", "QValue"});
    auto gq_c     = column(table, {"Global.Q.Value", "Global.Peptidoform.Q.Value"});
    auto pgq_c    = column(table, {"PG.Q.Value", "Protein.Q.Value", "Global.PG.Q.Value"});
    auto decoy_c  = column(table, {"Decoy"});
    auto pep_c    = column(table, {"PEP"});
    auto ev_c     = column(table, {"Evidence", "CScore"});
    auto ft_c     = column(table, {"Fragment.Type", "FragmentType"});
    auto fs_c     = column(table, {"Fragment.Series.Number", "FragmentSeriesNumber"});
    auto fz_c     = column(table, {"Fragment.Charge", "FragmentCharge"});

    if (!seq_c || !charge_c)
    { throw std::runtime_error("reference has no Modified.Sequence / Precursor.Charge column; it is not a DIA-NN report or library"); }

    // The gate contract (review M8). A gate is enabled when its threshold is
    // below 1. In report mode every enabled gate's column must exist; in
    // empirical-library mode a missing column is a recorded bypass.
    auto gate = [&](const char* name, double threshold, const std::shared_ptr<arrow::ChunkedArray>& c) -> bool
    {
      if (threshold >= 1.0) { return false; }
      if (c) { return true; }
      if (p.require_gates)
      { throw std::runtime_error(std::string("reference has no ") + name + " column but that gate is enabled; "
                                 "pass -empirical_library to declare a pre-filtered input and record the bypass"); }
      stats.gates_bypassed.emplace_back(name);
      return false;
    };
    const bool use_q = gate("Q.Value", p.q_precursor, q_c);
    const bool use_gq = gate("Global.Q.Value", p.q_global, gq_c);
    const bool use_pgq = gate("PG.Q.Value", p.q_protein, pgq_c);
    if (!decoy_c && p.require_gates)
    { throw std::runtime_error("reference has no Decoy column; pass -empirical_library if it is a target-only library"); }
    if (p.dedup == RefineParams::Dedup::HighestEvidence && !ev_c)
    { throw std::runtime_error("dedup=highest_evidence needs an Evidence (or CScore) column; the reference has none"); }
    const bool have_fragments = ft_c && fs_c && fz_c;
    if (p.min_fragments > 0 && !have_fragments)
    { throw std::runtime_error("min_fragments > 0 needs fragment identities (Fragment.Type/Series.Number/Charge); "
                               "a report carries none -- counting rows would count duplicates as fragments"); }

    const std::int64_t rows = table->num_rows();
    stats.ids_rows = static_cast<std::size_t>(rows);
    const auto seq = toStrings(seq_c, rows);
    const auto run = toStrings(run_c, rows);
    const auto charge = toDoubles(charge_c, rows);
    const auto rt = toDoubles(rt_c, rows), im = toDoubles(im_c, rows);
    const auto qv = toDoubles(q_c, rows), gq = toDoubles(gq_c, rows), pgq = toDoubles(pgq_c, rows);
    const auto decoy = toDoubles(decoy_c, rows), pep = toDoubles(pep_c, rows), ev = toDoubles(ev_c, rows);
    const auto ft = toStrings(ft_c, rows);
    const auto fs = toDoubles(fs_c, rows);
    const auto fz = toDoubles(fz_c, rows);

    // One run. A merged multi-run report mixes chromatographies; refuse it.
    if (run_c)
    {
      for (std::int64_t r = 0; r < rows; ++r)
      {
        if (run[static_cast<std::size_t>(r)].empty()) { continue; }
        if (stats.run.empty()) { stats.run = run[static_cast<std::size_t>(r)]; }
        else if (stats.run != run[static_cast<std::size_t>(r)])
        { throw std::runtime_error("reference contains more than one run (" + stats.run + ", " +
                                   run[static_cast<std::size_t>(r)] + "); refinement is per run"); }
      }
    }

    ObsMap out;
    out.reserve(static_cast<std::size_t>(rows) / 4 + 16);
    std::unordered_map<std::string, std::set<std::tuple<std::string, int, int>>> frag_ids;
    std::unordered_map<std::string, std::uint32_t> first_seen;
    std::set<std::string> distinct;

    for (std::int64_t r = 0; r < rows; ++r)
    {
      const std::size_t i = static_cast<std::size_t>(r);
      if (decoy_c)
      {
        if (!finite(decoy[i])) { ++stats.ids_q_invalid; continue; }
        if (decoy[i] != 0.0) { ++stats.ids_decoy; continue; }
      }
      const double zf = charge[i];
      if (!finite(zf) || zf < 1.0 || zf > 8.0 || zf != std::floor(zf)) { ++stats.ids_charge_invalid; continue; }
      const int z = static_cast<int>(zf);
      std::size_t unknown = 0;
      const std::string k = canonicalModifiedSequence(seq[i], &unknown) + "/" + std::to_string(z);
      stats.ids_unknown_mod_tokens += unknown;
      distinct.insert(k);

      // Gates. A missing or non-numeric q is not "passes"; it is rejected (M5/M8).
      if (use_q)   { if (!finite(qv[i]))  { ++stats.ids_q_invalid; continue; } if (qv[i]  > p.q_precursor) { ++stats.ids_q_above; continue; } }
      if (use_gq)  { if (!finite(gq[i]))  { ++stats.ids_q_invalid; continue; } if (gq[i]  > p.q_global)    { ++stats.ids_q_above; continue; } }
      if (use_pgq) { if (!finite(pgq[i])) { ++stats.ids_q_invalid; continue; } if (pgq[i] > p.q_protein)   { ++stats.ids_q_above; continue; } }

      if (have_fragments && finite(fs[i]) && finite(fz[i]))
      { frag_ids[k].insert(std::make_tuple(ft[i], static_cast<int>(fs[i]), static_cast<int>(fz[i]))); }

      Observation o;
      o.rt = static_cast<float>(finite(rt[i]) ? rt[i] : NaN);
      o.im = static_cast<float>(finite(im[i]) ? im[i] : NaN);
      o.q = static_cast<float>(finite(qv[i]) ? qv[i] : 1.0);
      o.pep = static_cast<float>(finite(pep[i]) ? pep[i] : 1.0);
      o.evidence = static_cast<float>(finite(ev[i]) ? ev[i] : 0.0);

      auto it = out.find(k);
      if (it == out.end()) { out.emplace(k, o); first_seen[k] = static_cast<std::uint32_t>(r); continue; }

      // Deterministic: strict improvement on the ranking quantity wins; ties keep
      // the first observation seen. Abundance is never a criterion (M9).
      bool better = false;
      if (p.dedup == RefineParams::Dedup::HighestEvidence) { better = o.evidence > it->second.evidence; }
      else { better = (o.q < it->second.q) || (o.q == it->second.q && o.pep < it->second.pep); }
      if (better) { it->second = o; }
    }

    for (auto& [k, o] : out)
    { if (auto f = frag_ids.find(k); f != frag_ids.end()) { o.fragments = static_cast<std::uint32_t>(f->second.size()); } }
    if (p.min_fragments > 0)
    {
      for (auto it = out.begin(); it != out.end();)
      { if (it->second.fragments < p.min_fragments) { it = out.erase(it); } else { ++it; } }
    }

    // Censoring against the INSTRUMENT limit the caller declared, never the data's maximum (M11).
    if (p.im_ramp_top > 0.0)
    {
      for (auto& [k, o] : out)
      {
        if (std::isfinite(o.im) && o.im > p.im_ramp_top - p.im_ramp_margin)
        { o.im = std::numeric_limits<float>::quiet_NaN(); ++stats.ids_ramp_censored; }
      }
    }

    stats.ids_precursors = distinct.size();
    stats.ids_passing = out.size();
    if (out.empty()) { throw std::runtime_error("no reference observation passed the gates; nothing to refine against"); }
    return out;
  }

  void LibraryRefiner::refine(Library& library, const ObsMap& obs,
                              const RefineParams& p, RefineStats& stats)
  {
    if (p.write_intensity)
    {
      throw std::runtime_error(
        "-write_intensity is not implemented. The paper that motivates this tool measures "
        "fragment-intensity replacement as a wash across three separate figures, so it is the one "
        "component with no evidence behind it. It is refused rather than stubbed so that no arm "
        "can silently run without it.");
    }
    if (p.rt_unit == RefineParams::RtUnit::MinMax && !p.filter)
    { throw std::runtime_error("rt_unit=minmax with the filter off would leave matched precursors on a 0..100 scale "
                               "and unmatched ones on the original scale in one library; refused"); }

    auto& pre = library.precursors();
    const std::size_t n = library.precursorCount();
    stats.library_before = n;

    std::vector<std::size_t> keep, matched_idx;
    keep.reserve(n);
    std::vector<double> rt_resid, im_resid;
    std::unordered_map<std::string, char> used;
    used.reserve(obs.size());

    for (std::size_t i = 0; i < n; ++i)
    {
      const std::string k = key(library.strings().get(pre.modified_sequence[i]), static_cast<int>(pre.charge[i]));
      auto it = obs.find(k);
      const bool is_decoy = pre.decoy[i] != 0;

      // A decoy is kept exactly when its key matched: ODIA's decoys carry the
      // TARGET's sequence with shifted fragments, so they share this key. Keeping
      // unmatched decoys "to preserve pairing" produced a 99.2%-decoy library once.
      if (it == obs.end()) { if (!p.filter) { keep.push_back(i); } continue; }
      keep.push_back(i);
      matched_idx.push_back(i);
      if (is_decoy) { ++stats.decoys_kept; continue; }

      used[k] = 1;
      ++stats.matched;
      if (p.write_rt && std::isfinite(it->second.rt) && std::isfinite(pre.irt[i]))
      { rt_resid.push_back(static_cast<double>(pre.irt[i]) - it->second.rt); }
      if (p.write_im && std::isfinite(it->second.im) && std::isfinite(pre.im[i]) && pre.charge[i] >= p.im_min_charge)
      { im_resid.push_back(static_cast<double>(pre.im[i]) - it->second.im); }
    }

    stats.ids_unmatched = obs.size() - used.size();
    stats.match_fraction = obs.empty() ? 0.0 : static_cast<double>(stats.matched) / static_cast<double>(obs.size());
    if (stats.matched == 0)
    { throw std::runtime_error("no reference precursor matched the library. That is a join failure -- check that both "
                               "sides use the same alkylation state and modification naming -- not an empty run."); }
    if (p.min_match_fraction > 0.0 && stats.match_fraction < p.min_match_fraction)
    { throw std::runtime_error("only " + std::to_string(stats.matched) + " of " + std::to_string(obs.size()) +
                               " reference precursors matched, below the required fraction"); }

    // Residuals BEFORE the write; after it they are zero by construction.
    auto summarise = [](std::vector<double>& v, double& m, double& s, double& p95, std::size_t& cnt)
    {
      cnt = v.size();
      m = mean(v); s = sd(v, m);
      std::vector<double> a; a.reserve(v.size());
      for (double x : v) { a.push_back(std::abs(x)); }
      p95 = quantile(std::move(a), 0.95);
    };
    summarise(rt_resid, stats.rt_resid_mean, stats.rt_resid_sd, stats.rt_resid_p95, stats.rt_resid_n);
    summarise(im_resid, stats.im_resid_mean, stats.im_resid_sd, stats.im_resid_p95, stats.im_resid_n);

    // MinMax needs a finite, non-degenerate observed range over the matched set.
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    if (p.rt_unit == RefineParams::RtUnit::MinMax)
    {
      for (std::size_t i : matched_idx)
      {
        const float r = obs.at(key(library.strings().get(pre.modified_sequence[i]), static_cast<int>(pre.charge[i]))).rt;
        if (std::isfinite(r)) { lo = std::min(lo, static_cast<double>(r)); hi = std::max(hi, static_cast<double>(r)); }
      }
      if (!(std::isfinite(lo) && std::isfinite(hi) && hi > lo))
      { throw std::runtime_error("rt_unit=minmax: the matched observations have no finite range to rescale"); }
    }

    for (std::size_t i : matched_idx)
    {
      const Observation& o = obs.at(key(library.strings().get(pre.modified_sequence[i]), static_cast<int>(pre.charge[i])));
      const bool is_decoy = pre.decoy[i] != 0;
      if (p.write_rt)
      {
        if (std::isfinite(o.rt))
        {
          double v = o.rt;
          if (p.rt_unit == RefineParams::RtUnit::MinMax) { v = 100.0 * (v - lo) / (hi - lo); }
          pre.irt[i] = static_cast<float>(v);
          ++stats.rt_written;
        }
        else if (!is_decoy) { ++stats.rt_missing; }   // left as predicted, and said so
      }
      if (p.write_im)
      {
        if (pre.charge[i] < p.im_min_charge) { if (!is_decoy) { ++stats.im_charge_excluded; } }
        else if (std::isfinite(o.im))
        {
          pre.im[i] = o.im;
          // CCS was derived from the PREDICTED 1/K0 and is now inconsistent with it.
          if (i < pre.ccs.size()) { pre.ccs[i] = std::numeric_limits<float>::quiet_NaN(); }
          ++stats.im_written;
        }
        else if (!is_decoy) { ++stats.im_missing; }
      }
    }

    if (p.filter && keep.size() != n) { library = library.subsetByIndex(keep); }
    stats.library_after = library.precursorCount();
  }
}
