// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// Two things the whole trainer rests on, tested against the stock ONNX:
///  1. PARITY: the libtorch transcription loaded from the ONNX predicts what
///     ONNX Runtime predicts (via DIALibGen's PeptDeepPredictor), for both
///     heads, on real peptides of several lengths with a modification.
///  2. ROUND TRIP: load -> store writes back a byte-identical file, and a
///     perturbed model written back and re-loaded returns the perturbation
///     (so the gate permutation is its own inverse and every tensor is
///     reached).
///
///   tune_parity <rt.onnx> <ccs.onnx>

#include <odia/tune/OnnxWeights.h>
#include <odia/PeptDeepEncoder.h>
#include <odia/PeptDeepPredictor.h>

#include <OpenMS/CHEMISTRY/AASequence.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace ODIA;
using namespace ODIA::tune;

namespace
{
  int failures = 0;
  void check(bool ok, const std::string& what)
  {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) { ++failures; }
  }

  std::vector<OpenMS::AASequence> peptides()
  {
    std::vector<OpenMS::AASequence> v;
    for (const char* s : {"PEPTIDEK", "LGEHNIDVLEGNEQFINAAK", "AC(UniMod:4)DEFGHIK", "M(UniMod:35)VLSDGK",
                          "TTPSYVAFTDTER", "QWERTYIPASDFGHK", "GLVLIAFSQYLQQC(UniMod:4)PFDEHVK", "SVAAAR"})
    { v.push_back(OpenMS::AASequence::fromString(s)); }
    return v;
  }

  /// Predict with the libtorch model, one length group at a time (as training does).
  std::vector<float> torchPredict(Head& model, const std::vector<OpenMS::AASequence>& peps, const std::vector<int>& charges)
  {
    torch::NoGradGuard ng;
    model->eval();
    std::vector<float> out(peps.size(), NAN);
    for (const auto& group : PeptDeepEncoder::groupByLength(peps))
    {
      std::vector<OpenMS::AASequence> gp; std::vector<int> gz;
      for (auto i : group) { gp.push_back(peps[i]); gz.push_back(charges[i]); }
      auto b = PeptDeepEncoder::encode(gp, gz, 30.0f, "Lumos");
      const auto rows = static_cast<std::int64_t>(b.rows), L = static_cast<std::int64_t>(b.sequence_length);
      auto aa = torch::from_blob(b.aa_indices.data(), {rows, L}, torch::kInt64).clone();
      auto mx = torch::from_blob(b.mod_x.data(), {rows, L, MOD_FEATURES}, torch::kFloat32).clone();
      auto ch = torch::from_blob(b.charges.data(), {rows, 1}, torch::kFloat32).clone();
      auto y = model->forward(aa, mx, ch).contiguous();
      for (std::size_t k = 0; k < group.size(); ++k) { out[group[k]] = y[static_cast<std::int64_t>(k)].item<float>(); }
    }
    return out;
  }

  double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
  {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
      if (!std::isfinite(d)) { return INFINITY; }   // a NaN anywhere is a failure, not a zero
      m = std::max(m, d);
    }
    return m;
  }

  void parity(const std::string& path, bool ccs)
  {
    const auto peps = peptides();
    std::vector<int> charges; for (std::size_t i = 0; i < peps.size(); ++i) { charges.push_back(2 + static_cast<int>(i % 3)); }
    OnnxFile f = OnnxFile::read(path);
    Head model(ccs);
    const std::size_t n = loadWeights(f, model);
    check(n == 21, std::string(ccs ? "ccs" : "rt") + ": 21 initializers loaded (" + std::to_string(n) + ")");
    auto ours = torchPredict(model, peps, charges);
    PeptDeepPredictor ort(path, /*prefer_gpu=*/false, /*intra_op_threads=*/1);
    auto ref = ccs ? ort.predictCCS(peps, charges) : ort.predictRT(peps);
    const double d = maxAbsDiff(ours, ref);
    // RT is rt_norm (~0..1), CCS is A^2 (~300-600): tolerances scaled accordingly
    const double tol = ccs ? 5e-3 : 1e-5;
    char buf[256]; std::snprintf(buf, sizeof buf, "%s parity vs ONNX Runtime: max |diff| = %.3g (tol %.0e)", ccs ? "ccs" : "rt", d, tol);
    check(d <= tol, buf);
    for (std::size_t i = 0; i < 2; ++i) { std::printf("       %-28s z%d torch %.6f ort %.6f\n", peps[i].toString().c_str(), charges[i], ours[i], ref[i]); }

    // Round trip 1: unchanged model writes back byte-identical
    OnnxFile g = f;
    storeWeights(g, model);
    check(g.bytes == f.bytes, std::string(ccs ? "ccs" : "rt") + ": store(load(x)) is byte-identical");

    // Round trip 2: perturb every trainable parameter, write back, reload into a fresh model, compare
    {
      torch::NoGradGuard ng;
      for (auto& p : model->parameters()) { p.add_(0.001 * torch::arange(p.numel(), torch::kFloat32).reshape(p.sizes()) / static_cast<double>(std::max<std::int64_t>(1, p.numel()))); }
    }
    OnnxFile h = f;
    storeWeights(h, model);
    Head again(ccs);
    loadWeights(h, again);
    double worst = 0;
    auto a = model->named_parameters(); auto b = again->named_parameters();
    for (const auto& kv : a)
    {
      const double w = (kv.value() - *b.find(kv.key())).abs().max().item<double>();
      worst = std::isfinite(w) ? std::max(worst, w) : INFINITY;
    }
    check(worst == 0.0, std::string(ccs ? "ccs" : "rt") + ": perturbed model survives store->load exactly (worst " + std::to_string(worst) + ")");
    check(h.bytes != f.bytes, std::string(ccs ? "ccs" : "rt") + ": perturbed bytes differ from the original");
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) { std::cerr << "usage: tune_parity <rt.onnx> <ccs.onnx>\n"; return 2; }
  torch::set_num_threads(1);
  try
  {
    parity(argv[1], false);
    parity(argv[2], true);
  }
  catch (const std::exception& e) { std::cerr << "exception: " << e.what() << "\n"; return 1; }
  std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
