// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <odia/LibraryRefiner.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

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

  bool parseFragmentInfo(std::string_view info, std::string_view quant, std::string_view corr,
                         std::vector<ObservedFragment>& out, std::size_t* bad_tokens)
  {
    // DIA-NN terminates every list with ';', so the last field is empty by
    // construction. Dropping exactly one such field keeps a genuinely empty
    // interior field visible as the length disagreement it is.
    auto split = [](std::string_view s)
    {
      std::vector<std::string_view> v;
      std::size_t a = 0;
      while (a <= s.size())
      {
        const std::size_t b = s.find(';', a);
        if (b == std::string_view::npos) { v.push_back(s.substr(a)); break; }
        v.push_back(s.substr(a, b - a));
        a = b + 1;
      }
      if (!v.empty() && v.back().empty()) { v.pop_back(); }
      return v;
    };
    // strtod, not from_chars: floating-point from_chars is still missing from
    // the libc++ some of the macOS runners ship.
    auto number = [](std::string_view t, double& v)
    {
      if (t.empty()) { return false; }
      const std::string z(t);
      char* end = nullptr;
      v = std::strtod(z.c_str(), &end);
      return end == z.c_str() + z.size() && std::isfinite(v);
    };

    const auto fi = split(info), fq = split(quant);
    const auto fc = corr.empty() ? std::vector<std::string_view>() : split(corr);
    if (fi.size() != fq.size() || (!corr.empty() && fc.size() != fi.size())) { return false; }

    out.clear();
    out.reserve(fi.size());
    for (std::size_t n = 0; n < fi.size(); ++n)
    {
      // <letter><ordinal>^<charge>/<mz>, by hand: a report has about a million
      // of these and the rest of this file is hand-parsed as well.
      const std::string_view t = fi[n];
      auto bad = [&] { if (bad_tokens) { ++*bad_tokens; } };
      std::size_t i = 0;
      while (i < t.size() && std::isalpha(static_cast<unsigned char>(t[i]))) { ++i; }
      // parseFragmentType reads only the first letter, so "by9" would resolve
      // to a b ion. A series this does not know is a bad token, not a b ion.
      if (i != 1) { bad(); continue; }
      const FragmentType type = parseFragmentType(t.substr(0, 1));
      if (type == FragmentType::Unknown) { bad(); continue; }

      std::size_t j = i;
      unsigned ordinal = 0;
      while (j < t.size() && std::isdigit(static_cast<unsigned char>(t[j])) && ordinal <= 255)
      { ordinal = ordinal * 10 + static_cast<unsigned>(t[j] - '0'); ++j; }
      if (j == i || ordinal == 0 || ordinal > 255 || j >= t.size() || t[j] != '^') { bad(); continue; }

      std::size_t k = ++j;
      unsigned z = 0;
      while (k < t.size() && std::isdigit(static_cast<unsigned char>(t[k])) && z <= 9) { z = z * 10 + static_cast<unsigned>(t[k] - '0'); ++k; }
      if (k == j || z == 0 || z > 4 || k >= t.size() || t[k] != '/') { bad(); continue; }

      double mz = 0.0, q = 0.0, c = std::numeric_limits<double>::quiet_NaN();
      if (!number(t.substr(k + 1), mz) || mz <= 0.0 || !number(fq[n], q)) { bad(); continue; }
      if (!corr.empty() && !number(fc[n], c)) { bad(); continue; }

      ObservedFragment f;
      f.type = type;
      f.ordinal = static_cast<std::uint8_t>(ordinal);
      f.charge = static_cast<std::int8_t>(z);
      f.mz = mz;
      f.quant = static_cast<float>(q);
      f.correlation = static_cast<float>(c);
      out.push_back(f);
    }
    return true;
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
    // A REPORT packs its fragments into three positionally paired strings; the
    // three columns above are what an empirical LIBRARY carries instead.
    auto finfo_c  = column(table, {"Fragment.Info"});
    auto fquant_c = column(table, {"Fragment.Quant.Raw"});
    auto fcorr_c  = column(table, {"Fragment.Correlations"});

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
    if (p.write_intensity && !(finfo_c && fquant_c))
    { throw std::runtime_error("-write_intensity needs the reference's Fragment.Info and Fragment.Quant.Raw columns; "
                               "DIA-NN writes them only with --report-lib-info. Refused rather than degraded to an "
                               "RT-only refinement that would pass for the treatment arm."); }
    if (p.write_intensity && !fcorr_c && p.intensity_min_correlation > -1.0)
    { throw std::runtime_error("-write_intensity needs Fragment.Correlations to apply its correlation gate; pass "
                               "-intensity_min_correlation -1 to replace intensities with no quality gate at all, "
                               "and have that recorded"); }
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
    // Three string columns of a 1.6M-row report come to roughly 400 MB. An
    // RT-only refinement must not pay that, so they are read only on request.
    const bool need_frags = p.write_intensity;
    const auto finfo = need_frags ? toStrings(finfo_c, rows) : std::vector<std::string>();
    const auto fquant = need_frags ? toStrings(fquant_c, rows) : std::vector<std::string>();
    const auto fcorr = (need_frags && fcorr_c) ? toStrings(fcorr_c, rows) : std::vector<std::string>();

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
      if (need_frags)
      {
        std::size_t bad = 0;
        if (!parseFragmentInfo(finfo[i], fquant[i], fcorr.empty() ? std::string_view{} : std::string_view(fcorr[i]),
                               o.frags, &bad))
        { ++stats.intensity_row_length_mismatch; o.frags.clear(); }
        stats.intensity_bad_tokens += bad;
      }
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

    if (need_frags && rows > 0 &&
        static_cast<double>(stats.intensity_row_length_mismatch) / static_cast<double>(rows) > 0.01)
    { throw std::runtime_error("Fragment.Info, Fragment.Quant.Raw and Fragment.Correlations disagree in length on " +
                               std::to_string(stats.intensity_row_length_mismatch) + " rows; the columns are not "
                               "positionally paired, and every intensity would land on the wrong fragment"); }

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

  namespace
  {
    /// (type, ordinal, |charge|). Neutral loss is NOT part of the identity: the
    /// report has no loss field, so a loss-bearing transition is unmatchable by
    /// construction and is counted as such rather than matched to its no-loss twin.
    std::uint32_t ident(FragmentType t, unsigned ordinal, unsigned z)
    { return (static_cast<std::uint32_t>(t) << 16) | (ordinal << 8) | z; }

    /// The ONE definition of "trusted", so the gate and its counters cannot drift apart.
    enum class Trust { Yes, ZeroQuant, Correlation };
    Trust trust(const ObservedFragment& f, const RefineParams& p)
    {
      if (!(f.quant > 0.0f)) { return Trust::ZeroQuant; }
      if (p.intensity_min_correlation > -1.0 && !(f.correlation > p.intensity_min_correlation)) { return Trust::Correlation; }
      return Trust::Yes;
    }

    struct Kept { std::uint32_t j; float value; };   ///< source transition, and the intensity it will carry

    void applyObservedIntensities(Library& library, const LibraryRefiner::ObsMap& obs,
                                  const std::vector<std::size_t>& matched_idx,
                                  const RefineParams& p, RefineStats& stats)
    {
      auto& pre = library.precursors();
      auto& t = library.transitions();

      // Targets and their decoys share a key (a decoy row carries its TARGET's
      // sequence), and must leave here with the same fragments and the same
      // pattern: a target-only change is the anti-conservative FDR mode.
      struct Group { std::vector<std::size_t> targets, decoys; };
      std::unordered_map<std::string, Group> groups;
      groups.reserve(matched_idx.size());
      for (std::size_t i : matched_idx)
      {
        Group& g = groups[LibraryRefiner::key(library.strings().get(pre.modified_sequence[i]), static_cast<int>(pre.charge[i]))];
        (pre.decoy[i] != 0 ? g.decoys : g.targets).push_back(i);
      }

      std::unordered_map<std::size_t, std::vector<Kept>> plan;   // precursor -> its new transition list
      std::size_t rank_agree = 0, n_before = 0, n_after = 0;

      for (const auto& [k, g] : groups)
      {
        if (g.targets.empty()) { continue; }
        // Two targets on one key is a library defect. Guessing which of them
        // owns the observation is how a feature becomes plausibly wrong.
        if (g.targets.size() > 1) { ++stats.intensity_duplicate_key; continue; }
        const std::size_t i = g.targets.front();
        const Observation& o = obs.at(k);
        if (o.frags.empty()) { continue; }
        ++stats.intensity_candidate_precursors;

        std::unordered_map<std::uint32_t, const ObservedFragment*> seen;
        seen.reserve(o.frags.size());
        for (const ObservedFragment& f : o.frags)
        {
          const auto [it, fresh] = seen.emplace(ident(f.type, f.ordinal, static_cast<unsigned>(f.charge)), &f);
          (void)it;
          if (!fresh) { ++stats.intensity_bad_tokens; }   // the same fragment twice in one row: keep the first
        }

        const std::uint32_t b = pre.transition_begin[i], c = pre.transition_count[i];
        stats.intensity_candidate_transitions += c;
        float lib_max = 0.0f;
        std::uint32_t lib_top = 0;
        std::unordered_set<std::uint32_t> in_library;
        std::vector<Kept> kept;
        std::unordered_map<std::uint32_t, float> kept_by_ident;
        for (std::uint32_t j = b; j < b + c; ++j)
        {
          if (t.library_intensity[j] > lib_max) { lib_max = t.library_intensity[j]; }
          if (t.loss[j] != LossType::None) { ++stats.intensity_loss_bearing; continue; }
          // The file's 0 placeholder means 1; DIALibGen's decoy stage reads it so too.
          const unsigned z = t.charge[j] == 0 ? 1u : static_cast<unsigned>(std::abs(static_cast<int>(t.charge[j])));
          const std::uint32_t id = ident(t.type[j], t.ordinal[j], z);
          in_library.insert(id);
          if (t.library_intensity[j] == lib_max) { lib_top = id; }
          const auto f = seen.find(id);
          if (f == seen.end()) { ++stats.intensity_unmatched_in_library; continue; }
          // Identity agreeing is not enough. A shifted ordinal convention, a
          // mis-parse, or an -in that is not the searched library all agree on
          // identity for SOME fragment and put every intensity on the wrong one.
          const double lib_mz = fromFixed(t.product_mz[j]);
          if (std::abs(f->second->mz - lib_mz) > p.intensity_mz_tol_ppm * f->second->mz * 1e-6)
          { ++stats.intensity_mz_mismatch; continue; }
          ++stats.intensity_matched_transitions;
          switch (trust(*f->second, p))
          {
            case Trust::ZeroQuant:   ++stats.intensity_gated_zero_quant; break;
            case Trust::Correlation: ++stats.intensity_gated_correlation; break;
            case Trust::Yes:         kept.push_back({j, f->second->quant}); break;
          }
        }
        // Never ADDED. A target-only fragment has no decoy counterpart to add:
        // the decoy's m/z must come from the decoy's own sequence, and the
        // library stores the target's sequence on the decoy row.
        for (const auto& [id, f] : seen) { (void)f; if (!in_library.count(id)) { ++stats.intensity_observed_not_in_library; } }

        float obs_max = 0.0f;
        for (const Kept& x : kept) { obs_max = std::max(obs_max, x.value); }
        // The floor is relative to the KEPT set's maximum. The generator's is
        // relative to the full model spectrum's peak, which does not exist here.
        const std::size_t unfloored = kept.size();
        kept.erase(std::remove_if(kept.begin(), kept.end(), [&](const Kept& x)
                   { return static_cast<double>(x.value) < p.intensity_min_relative * static_cast<double>(obs_max); }),
                   kept.end());
        stats.intensity_gated_floor += unfloored - kept.size();

        // restrict off: all or nothing, so transition counts cannot move.
        // kept.empty() is tested on its own: -intensity_min_fragments 0 must not reach a division by obs_max = 0.
        if (kept.empty() || (!p.intensity_restrict && kept.size() != c) || kept.size() < p.intensity_min_fragments)
        { ++stats.intensity_kept_predicted; continue; }

        double scale = 1.0;
        switch (p.intensity_norm)
        {
          case RefineParams::IntensityNorm::LibraryMax: scale = (lib_max > 0.0f ? static_cast<double>(lib_max) : 1.0) / obs_max; break;
          case RefineParams::IntensityNorm::BasePeak:   scale = 1.0 / obs_max; break;
          case RefineParams::IntensityNorm::Sum:
          { double sum = 0.0; for (const Kept& x : kept) { sum += x.value; } scale = 1.0 / sum; break; }
          case RefineParams::IntensityNorm::Raw: break;
        }
        std::uint32_t obs_top = 0;
        for (Kept& x : kept)
        {
          if (x.value == obs_max)
          { obs_top = ident(t.type[x.j], t.ordinal[x.j], t.charge[x.j] == 0 ? 1u : static_cast<unsigned>(std::abs(static_cast<int>(t.charge[x.j])))); }
          x.value = static_cast<float>(static_cast<double>(x.value) * scale);
          kept_by_ident[ident(t.type[x.j], t.ordinal[x.j], t.charge[x.j] == 0 ? 1u : static_cast<unsigned>(std::abs(static_cast<int>(t.charge[x.j]))))] = x.value;
        }

        // INVARIANT: a decoy never meets the report. Its fragment m/z were
        // recomputed from the shuffled sequence and are not the target's, so an
        // m/z check on a decoy would refuse every real library. A decoy takes the
        // TARGET's value by fragment identity -- which is what DIALibGen's own
        // decoy stage does -- or the whole key keeps its predictions.
        std::vector<std::pair<std::size_t, std::vector<Kept>>> decoy_plans;
        bool symmetric = true;
        for (std::size_t d : g.decoys)
        {
          const std::uint32_t db = pre.transition_begin[d], dc = pre.transition_count[d];
          std::vector<Kept> dk;
          for (std::uint32_t j = db; j < db + dc; ++j)
          {
            if (t.loss[j] != LossType::None) { continue; }
            const unsigned z = t.charge[j] == 0 ? 1u : static_cast<unsigned>(std::abs(static_cast<int>(t.charge[j])));
            if (const auto v = kept_by_ident.find(ident(t.type[j], t.ordinal[j], z)); v != kept_by_ident.end())
            { dk.push_back({j, v->second}); }
          }
          if ((!p.intensity_restrict && dk.size() != dc) || dk.size() < p.intensity_min_fragments) { symmetric = false; break; }
          decoy_plans.emplace_back(d, std::move(dk));
        }
        if (!symmetric) { ++stats.intensity_decoy_asymmetry; ++stats.intensity_kept_predicted; continue; }

        if (obs_top == lib_top) { ++rank_agree; }
        n_before += c;
        n_after += kept.size();
        ++stats.intensity_replaced_precursors;
        stats.intensity_replaced_decoys += decoy_plans.size();
        plan.emplace(i, std::move(kept));
        for (auto& [d, dk] : decoy_plans) { plan.emplace(d, std::move(dk)); }
      }

      const std::size_t judged = stats.intensity_matched_transitions + stats.intensity_mz_mismatch;
      stats.intensity_mz_mismatch_fraction = judged ? static_cast<double>(stats.intensity_mz_mismatch) / static_cast<double>(judged) : 0.0;
      if (stats.intensity_mz_mismatch_fraction > p.intensity_max_mz_mismatch)
      { throw std::runtime_error(std::to_string(100.0 * stats.intensity_mz_mismatch_fraction) + "% of identity-matched fragments "
                                 "disagree with the library's m/z by more than " + std::to_string(p.intensity_mz_tol_ppm) +
                                 " ppm -- -in is probably not the library this run was searched against, or the fragment "
                                 "numbering differs. Refused; raise -intensity_max_mz_mismatch to survey it instead."); }
      if (stats.intensity_replaced_precursors == 0)
      { throw std::runtime_error("-write_intensity replaced no precursor. That is a join or parse failure, not a run in "
                                 "which nothing was observed, and the output would be identical to the control arm."); }
      stats.intensity_rank_agreement = static_cast<double>(rank_agree) / static_cast<double>(stats.intensity_replaced_precursors);
      stats.intensity_transitions_before = static_cast<double>(n_before) / static_cast<double>(stats.intensity_replaced_precursors);
      stats.intensity_transitions_after = static_cast<double>(n_after) / static_cast<double>(stats.intensity_replaced_precursors);

      // A precursor's transition count changes, so every later offset moves and
      // nothing can be edited in place: rebuild, as DIALibGen's own intensity
      // stage does. The transient cost is one extra copy of the transition arrays.
      Library::TransitionArrays built;
      const std::size_t total = t.product_mz.size();
      built.product_mz.reserve(total); built.library_intensity.reserve(total); built.type.reserve(total);
      built.ordinal.reserve(total); built.charge.reserve(total); built.loss.reserve(total);
      auto copy = [&](std::uint32_t j, float intensity)
      {
        built.product_mz.push_back(t.product_mz[j]); built.library_intensity.push_back(intensity);
        built.type.push_back(t.type[j]); built.ordinal.push_back(t.ordinal[j]);
        built.charge.push_back(t.charge[j]); built.loss.push_back(t.loss[j]);
      };
      const std::size_t n = library.precursorCount();
      std::vector<std::uint32_t> begin(n), count(n);
      for (std::size_t i = 0; i < n; ++i)
      {
        begin[i] = static_cast<std::uint32_t>(built.product_mz.size());
        const auto it = plan.find(i);
        if (it == plan.end())
        { for (std::uint32_t j = pre.transition_begin[i]; j < pre.transition_begin[i] + pre.transition_count[i]; ++j) { copy(j, t.library_intensity[j]); } }
        else
        {
          std::vector<Kept>& v = it->second;
          if (p.intensity_rerank)
          {
            // Ties break on identity, deliberately NOT on m/z: a decoy's m/z
            // differ from its target's, and an m/z tie-break would order a decoy
            // differently from the target whose pattern it has to mirror.
            std::stable_sort(v.begin(), v.end(), [&](const Kept& a, const Kept& b)
            {
              if (a.value != b.value) { return a.value > b.value; }
              return std::make_tuple(t.type[a.j], t.ordinal[a.j], t.charge[a.j]) < std::make_tuple(t.type[b.j], t.ordinal[b.j], t.charge[b.j]);
            });
          }
          for (const Kept& x : v) { copy(x.j, x.value); }
        }
        count[i] = static_cast<std::uint32_t>(built.product_mz.size()) - begin[i];
      }
      t = std::move(built);
      pre.transition_begin = std::move(begin);
      pre.transition_count = std::move(count);
      library.shrinkToFit();
    }
  }

  void LibraryRefiner::refine(Library& library, const ObsMap& obs,
                              const RefineParams& p, RefineStats& stats)
  {
    if (p.write_intensity && !p.filter && !p.allow_mixed_intensity)
    { throw std::runtime_error("-write_intensity with the filter off would leave matched precursors carrying the run's "
                               "OBSERVED intensities and unmatched ones carrying MS2-model predictions in one library; "
                               "pass -allow_mixed_intensity to accept that and have it recorded"); }
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

    if (p.write_intensity) { applyObservedIntensities(library, obs, matched_idx, p, stats); }

    if (p.filter && keep.size() != n) { library = library.subsetByIndex(keep); }
    stats.library_after = library.precursorCount();
  }
}
