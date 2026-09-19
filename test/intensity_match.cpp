// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

// Fragment-intensity replacement, tested without any fixture: the library and
// the observations are built here, so every expected value is visible beside
// the assertion that checks it.
//
// The test that matters most is mz_mismatch_throws. A fragment numbering that
// is off by one still agrees on IDENTITY for most fragments, every counter
// still looks healthy, and every intensity lands on the wrong fragment. Only
// the m/z cross-check notices.

#include <odia/LibraryRefiner.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ODIA;

namespace
{
  int failures = 0;
  void check(bool ok, const std::string& what)
  { if (!ok) { ++failures; std::cerr << "FAIL: " << what << "\n"; } }

  struct Frag { FragmentType type; int ordinal; int charge; double mz; float intensity; };

  /// One target precursor "PEPTIDEK"/2 with the given transitions.
  Library makeLibrary(const std::vector<Frag>& frags)
  {
    Library lib;
    auto& p = lib.precursors();
    auto& t = lib.transitions();
    p.mz.push_back(toFixed(500.25)); p.irt.push_back(0.5f); p.im.push_back(1.0f); p.ccs.push_back(400.0f);
    p.charge.push_back(2); p.decoy.push_back(0);
    p.modified_sequence.push_back(lib.strings().intern("PEPTIDEK"));
    p.protein_group.push_back(lib.strings().intern("sp|P1|TEST"));
    p.transition_begin.push_back(0);
    p.transition_count.push_back(static_cast<std::uint32_t>(frags.size()));
    for (const Frag& f : frags)
    {
      t.product_mz.push_back(toFixed(f.mz)); t.library_intensity.push_back(f.intensity);
      t.type.push_back(f.type); t.ordinal.push_back(static_cast<std::uint8_t>(f.ordinal));
      t.charge.push_back(static_cast<std::int8_t>(f.charge)); t.loss.push_back(LossType::None);
    }
    return lib;
  }

  LibraryRefiner::ObsMap observe(const std::string& info, const std::string& quant, const std::string& corr)
  {
    Observation o;
    o.rt = 42.0f; o.im = 0.9f; o.q = 0.001f;
    check(parseFragmentInfo(info, quant, corr, o.frags), "fixture triple parses");
    LibraryRefiner::ObsMap m;
    m.emplace(LibraryRefiner::key("PEPTIDEK", 2), o);
    return m;
  }

  const std::vector<Frag> LIB = {
    {FragmentType::Y, 5, 1, 600.30, 1.00f}, {FragmentType::Y, 4, 1, 500.25, 0.80f},
    {FragmentType::B, 3, 1, 300.15, 0.60f}, {FragmentType::Y, 6, 1, 700.35, 0.40f},
  };

  void parser()
  {
    std::vector<ObservedFragment> f;
    std::size_t bad = 0;
    check(parseFragmentInfo("b9^1/668.37;y7^2/330.17;", "10;20;", "0.9;-0.1;", f, &bad) && f.size() == 2 && bad == 0,
          "a well-formed triple parses, and the trailing ';' makes no phantom fragment");
    check(f.size() == 2 && f[1].type == FragmentType::Y && f[1].ordinal == 7 && f[1].charge == 2 && f[1].correlation < 0,
          "type, ordinal, charge and a NEGATIVE correlation survive");
    f.assign(1, ObservedFragment{});
    check(!parseFragmentInfo("b9^1/668.37;y7^1/659.35;", "10;", "0.9;0.8;", f) && f.size() == 1,
          "ragged lists are a ROW failure and write nothing");
    bad = 0;
    check(parseFragmentInfo("by9^1/668.37;b0^1/1.0;b3^9/1.0;b3^1/x;y7^1/659.35;", "1;1;1;1;5;", "1;1;1;1;1;", f, &bad)
          && f.size() == 1 && bad == 4, "an unknown series, ordinal 0, charge 9 and a non-numeric m/z are bad tokens, not b ions");
    check(parseFragmentInfo("y7^1/659.35;", "5;", "", f) && f.size() == 1 && std::isnan(f[0].correlation),
          "an absent correlation column gives NaN, not zero");
  }

  void replaces_and_reranks()
  {
    Library lib = makeLibrary(LIB);
    // y4 is the observed base peak; b3 anti-correlates; y6 has no area.
    const auto obs = observe("y5^1/600.30;y4^1/500.25;b3^1/300.15;y6^1/700.35;", "500;1000;800;0;", "0.9;0.95;-0.2;0.9;");
    RefineParams p; p.write_rt = false; p.write_intensity = true; p.intensity_min_fragments = 2;
    RefineStats st;
    LibraryRefiner::refine(lib, obs, p, st);
    const auto& t = lib.transitions();
    check(st.intensity_replaced_precursors == 1 && lib.precursors().transition_count[0] == 2, "two trusted fragments survive");
    check(st.intensity_gated_correlation == 1 && st.intensity_gated_zero_quant == 1, "each gate reports its own rejection");
    check(t.type[0] == FragmentType::Y && t.ordinal[0] == 4 && t.ordinal[1] == 5, "re-ranked by OBSERVED intensity: y4 now leads");
    // library_max: the observed maximum takes the value the precursor's maximum already had (1.0).
    check(std::abs(t.library_intensity[0] - 1.0f) < 1e-6f && std::abs(t.library_intensity[1] - 0.5f) < 1e-6f,
          "library_max scaling: 1000 -> 1.0 and 500 -> 0.5");
    check(std::abs(fromFixed(t.product_mz[0]) - 500.25) < 1e-3, "a transition keeps its OWN m/z when it moves");
    check(st.intensity_matched_transitions + st.intensity_mz_mismatch + st.intensity_unmatched_in_library +
          st.intensity_loss_bearing == st.intensity_candidate_transitions, "the four fates of a library transition are exhaustive");
    check(st.intensity_rank_agreement == 0.0, "rank agreement records that the predicted base peak (y5) was wrong here");
  }

  void no_restrict_preserves_counts()
  {
    Library lib = makeLibrary(LIB);
    const auto obs = observe("y5^1/600.30;y4^1/500.25;b3^1/300.15;y6^1/700.35;", "500;1000;800;0;", "0.9;0.95;-0.2;0.9;");
    RefineParams p; p.write_rt = false; p.write_intensity = true; p.intensity_restrict = false;
    RefineStats st;
    bool threw = false;
    // Not every transition is trusted, so under no-restrict this precursor keeps its
    // predictions -- and then NOTHING was replaced, which is an error by design.
    try { LibraryRefiner::refine(lib, obs, p, st); } catch (const std::runtime_error&) { threw = true; }
    check(threw && lib.precursors().transition_count[0] == 4 && lib.transitions().library_intensity[0] == 1.00f,
          "no-restrict never changes a transition count, and a run that replaces nothing is refused");
  }

  void mz_mismatch_throws()
  {
    Library lib = makeLibrary(LIB);
    // The same fragments, numbered one higher: identities still collide with the
    // library's (y5, y6 exist on both sides) but the m/z belong to other fragments.
    const auto obs = observe("y6^1/600.30;y5^1/500.25;b4^1/300.15;y7^1/700.35;", "500;1000;800;300;", "0.9;0.95;0.9;0.9;");
    RefineParams p; p.write_rt = false; p.write_intensity = true; p.intensity_min_fragments = 1;
    RefineStats st;
    std::string msg;
    try { LibraryRefiner::refine(lib, obs, p, st); } catch (const std::runtime_error& e) { msg = e.what(); }
    check(msg.find("ppm") != std::string::npos, "a shifted fragment numbering is REFUSED, naming the ppm check: " + msg);
    check(lib.transitions().library_intensity[0] == 1.00f, "and nothing was written before refusing");
  }

  void mixed_provenance_refused()
  {
    Library lib = makeLibrary(LIB);
    const auto obs = observe("y5^1/600.30;", "500;", "0.9;");
    RefineParams p; p.write_intensity = true; p.filter = false;
    RefineStats st;
    bool threw = false;
    try { LibraryRefiner::refine(lib, obs, p, st); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "-write_intensity with the filter off is refused without -allow_mixed_intensity");
  }
}

int main()
{
  parser();
  replaces_and_reranks();
  no_restrict_preserves_counts();
  mz_mismatch_throws();
  mixed_provenance_refused();
  if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
  std::cout << "intensity_match: ok\n";
  return 0;
}
