// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <odia/tune/OnnxWeights.h>

#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace ODIA::tune
{
  namespace
  {
    struct Walker
    {
      const std::vector<std::uint8_t>& b;
      std::uint64_t varint(std::size_t& pos) const
      {
        std::uint64_t r = 0; int sh = 0;
        for (;;)
        {
          if (pos >= b.size()) { throw std::runtime_error("truncated ONNX (varint)"); }
          if (sh > 63) { throw std::runtime_error("malformed ONNX (varint longer than 10 bytes)"); }
          const std::uint8_t x = b[pos++];
          r |= static_cast<std::uint64_t>(x & 0x7f) << sh; sh += 7;
          if (!(x & 0x80)) { return r; }
        }
      }
      /// Visit every field in [pos, end): cb(field, wire_type, value_or_offset, length).
      void walk(std::size_t pos, std::size_t end,
                const std::function<void(int, int, std::uint64_t, std::size_t, std::size_t)>& cb) const
      {
        while (pos < end)
        {
          const std::uint64_t key = varint(pos);
          const int field = static_cast<int>(key >> 3), wt = static_cast<int>(key & 7);
          if (wt == 0) { const std::uint64_t v = varint(pos); cb(field, wt, v, 0, 0); }
          else if (wt == 1) { if (end - pos < 8) { throw std::runtime_error("truncated ONNX (fixed64)"); } cb(field, wt, 0, pos, 8); pos += 8; }
          else if (wt == 5) { if (end - pos < 4) { throw std::runtime_error("truncated ONNX (fixed32)"); } cb(field, wt, 0, pos, 4); pos += 4; }
          else if (wt == 2)
          {
            const std::uint64_t ln = varint(pos);
            if (ln > end - pos) { throw std::runtime_error("truncated ONNX (length-delimited field runs past its message)"); }
            cb(field, wt, 0, pos, static_cast<std::size_t>(ln)); pos += static_cast<std::size_t>(ln);
          }
          else { throw std::runtime_error("ONNX: unsupported wire type"); }
        }
      }
      std::string str(std::size_t off, std::size_t len) const { return std::string(reinterpret_cast<const char*>(b.data() + off), len); }
    };
  }

  OnnxFile OnnxFile::read(const std::string& path)
  {
    OnnxFile f;
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open ONNX file: " + path); }
    f.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    Walker w{f.bytes};
    w.walk(0, f.bytes.size(), [&](int field, int wt, std::uint64_t, std::size_t off, std::size_t len)
    {
      if (field != 7 || wt != 2) { return; }                   // ModelProto.graph
      w.walk(off, off + len, [&](int gf, int gwt, std::uint64_t, std::size_t goff, std::size_t glen)
      {
        if (gwt != 2) { return; }
        if (gf == 1)                                             // GraphProto.node
        {
          Node n;
          w.walk(goff, goff + glen, [&](int nf, int nwt, std::uint64_t, std::size_t noff, std::size_t nlen)
          {
            if (nwt != 2) { return; }
            if (nf == 1) { n.inputs.push_back(w.str(noff, nlen)); }
            else if (nf == 4) { n.op_type = w.str(noff, nlen); }
          });
          f.nodes.push_back(std::move(n));
        }
        else if (gf == 5)                                        // GraphProto.initializer
        {
          Tensor t;
          w.walk(goff, goff + glen, [&](int tf, int twt, std::uint64_t v, std::size_t toff, std::size_t tlen)
          {
            if (tf == 1 && twt == 0) { t.dims.push_back(static_cast<std::int64_t>(v)); }
            else if (tf == 1 && twt == 2)                        // packed dims
            { std::size_t p = toff; while (p < toff + tlen) { t.dims.push_back(static_cast<std::int64_t>(w.varint(p))); } }
            else if (tf == 2 && twt == 0) { t.data_type = static_cast<std::int32_t>(v); }
            else if (tf == 8 && twt == 2) { t.name = w.str(toff, tlen); }
            else if (tf == 9 && twt == 2) { t.raw_offset = toff; t.raw_length = tlen; }
            else if (tf == 4 && twt == 2) { throw std::runtime_error("ONNX initializer uses float_data, not raw_data: " + t.name); }
          });
          f.initializers.push_back(std::move(t));
        }
        else if (gf == 11)                                       // GraphProto.input (ValueInfoProto)
        {
          w.walk(goff, goff + glen, [&](int vf, int vwt, std::uint64_t, std::size_t voff, std::size_t vlen)
          { if (vf == 1 && vwt == 2) { f.graph_inputs.push_back(w.str(voff, vlen)); } });
        }
      });
    });
    if (f.initializers.empty()) { throw std::runtime_error("no initializers in " + path); }
    return f;
  }

  void OnnxFile::write(const std::string& path) const
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) { throw std::runtime_error("cannot write " + path); }
  }

  bool OnnxFile::hasInitializer(const std::string& name) const
  { for (const auto& t : initializers) { if (t.name == name) { return true; } } return false; }

  const OnnxFile::Tensor& OnnxFile::initializer(const std::string& name) const
  {
    for (const auto& t : initializers) { if (t.name == name) { return t; } }
    throw std::runtime_error("ONNX has no initializer named " + name);
  }

  namespace
  {
    torch::Tensor asTensor(const OnnxFile& f, const OnnxFile::Tensor& t)
    {
      if (t.data_type != 1) { throw std::runtime_error("initializer is not float32: " + t.name); }
      std::uint64_t n = 1;
      for (auto d : t.dims)
      {
        if (d < 0 || (d > 0 && n > (std::uint64_t{1} << 40) / static_cast<std::uint64_t>(d))) { throw std::runtime_error("implausible initializer shape: " + t.name); }
        n *= static_cast<std::uint64_t>(d);
      }
      if (n * 4 != t.raw_length || t.raw_offset + t.raw_length > f.bytes.size()) { throw std::runtime_error("raw_data length mismatch: " + t.name); }
      auto out = torch::empty(t.dims, torch::kFloat32);
      std::memcpy(out.data_ptr<float>(), f.bytes.data() + t.raw_offset, t.raw_length);
      return out;
    }

    void putTensor(OnnxFile& f, const OnnxFile::Tensor& t, const torch::Tensor& src)
    {
      auto c = src.detach().to(torch::kCPU, torch::kFloat32).contiguous();
      if (static_cast<std::size_t>(c.numel()) * 4 != t.raw_length)
      { throw std::runtime_error("write-back size mismatch for " + t.name); }
      std::memcpy(f.bytes.data() + t.raw_offset, c.data_ptr<float>(), t.raw_length);
    }

    /// torch gate rows [i,f,g,o] -> ONNX [i,o,f,c]
    torch::Tensor gatesToOnnx(const torch::Tensor& w)
    { auto b = w.chunk(4, 0); return torch::cat({b[0], b[3], b[1], b[2]}, 0); }
    /// ONNX [i,o,f,c] -> torch [i,f,g,o]
    torch::Tensor gatesToTorch(const torch::Tensor& w)
    { auto b = w.chunk(4, 0); return torch::cat({b[0], b[2], b[3], b[1]}, 0); }

    struct Names
    {
      std::vector<std::string> lstm_w, lstm_r, lstm_b;   // per layer
      std::vector<std::string> matmul;                   // in graph order
    };
    Names positional(const OnnxFile& f)
    {
      Names n;
      for (const auto& node : f.nodes)
      {
        if (node.op_type == "LSTM" && node.inputs.size() >= 4)
        { n.lstm_w.push_back(node.inputs[1]); n.lstm_r.push_back(node.inputs[2]); n.lstm_b.push_back(node.inputs[3]); }
        else if (node.op_type == "MatMul" && node.inputs.size() == 2 && f.hasInitializer(node.inputs[1]))
        { n.matmul.push_back(node.inputs[1]); }
      }
      if (n.lstm_w.size() != 2) { throw std::runtime_error("expected 2 LSTM nodes, found " + std::to_string(n.lstm_w.size())); }
      if (n.matmul.size() != 2) { throw std::runtime_error("expected 2 MatMul constants, found " + std::to_string(n.matmul.size())); }
      return n;
    }

    /// The name-matched tensors: {torch parameter name, ONNX initializer name}.
    std::vector<std::pair<std::string, std::string>> named(const Head& model)
    {
      const std::string enc = model->ccs ? "ccs_encoder" : "rt_encoder";
      const std::string dec = model->ccs ? "ccs_decoder" : "rt_decoder";
      return {
        {"encoder.cnn_short.weight", enc + ".input_cnn.cnn_short.weight"}, {"encoder.cnn_short.bias", enc + ".input_cnn.cnn_short.bias"},
        {"encoder.cnn_medium.weight", enc + ".input_cnn.cnn_medium.weight"}, {"encoder.cnn_medium.bias", enc + ".input_cnn.cnn_medium.bias"},
        {"encoder.cnn_long.weight", enc + ".input_cnn.cnn_long.weight"}, {"encoder.cnn_long.bias", enc + ".input_cnn.cnn_long.bias"},
        {"encoder.rnn_h0", enc + ".hidden_nn.rnn_h0"}, {"encoder.rnn_c0", enc + ".hidden_nn.rnn_c0"},
        {"dec0.weight", dec + ".nn.0.weight"}, {"dec0.bias", dec + ".nn.0.bias"},
        {"prelu.weight", dec + ".nn.1.weight"},
        {"dec2.weight", dec + ".nn.2.weight"}, {"dec2.bias", dec + ".nn.2.bias"},
      };
    }

    /// A handle to the named parameter (shares storage; copy_ on it edits the model).
    torch::Tensor param(Head& model, const std::string& name)
    {
      auto params = model->named_parameters();
      auto* p = params.find(name);
      if (!p) { throw std::runtime_error("model has no parameter " + name); }
      return *p;
    }
  }

  std::size_t loadWeights(const OnnxFile& f, Head& model)
  {
    torch::NoGradGuard ng;
    std::size_t count = 0;
    for (const auto& [tname, oname] : named(model))
    {
      auto src = asTensor(f, f.initializer(oname));
      auto dst = param(model, tname);
      if (src.sizes() != dst.sizes()) { throw std::runtime_error("shape mismatch " + oname + " vs " + tname); }
      dst.copy_(src); ++count;
    }
    const Names n = positional(f);
    // mod_nn (2,103) is stored transposed (103,2); attn (1,256) as (256,1)
    auto copyExact = [&](const std::string& tname, const torch::Tensor& src)
    {
      auto dst = param(model, tname);
      if (src.sizes() != dst.sizes()) { throw std::runtime_error("shape mismatch loading " + tname + ": ONNX " + std::to_string(src.numel()) + " values vs model " + std::to_string(dst.numel())); }
      dst.copy_(src);
    };
    copyExact("encoder.mod_nn.weight", asTensor(f, f.initializer(n.matmul[0])).t()); ++count;
    copyExact("encoder.attn.weight", asTensor(f, f.initializer(n.matmul[1])).t()); ++count;
    for (int l = 0; l < 2; ++l)
    {
      auto W = asTensor(f, f.initializer(n.lstm_w[l]));   // (2, 512, in)
      auto R = asTensor(f, f.initializer(n.lstm_r[l]));   // (2, 512, 128)
      auto B = asTensor(f, f.initializer(n.lstm_b[l]));   // (2, 1024)
      if (W.dim() != 3 || W.size(0) != 2 || W.size(1) != 4 * HIDDEN || R.sizes() != torch::IntArrayRef({2, 4 * HIDDEN, HIDDEN}) || B.sizes() != torch::IntArrayRef({2, 8 * HIDDEN}))
      { throw std::runtime_error("LSTM layer " + std::to_string(l) + " has an unexpected W/R/B shape"); }
      for (int d = 0; d < 2; ++d)
      {
        const std::string sfx = "_l" + std::to_string(l) + (d ? "_reverse" : "");
        copyExact("encoder.rnn.weight_ih" + sfx, gatesToTorch(W[d]));
        copyExact("encoder.rnn.weight_hh" + sfx, gatesToTorch(R[d]));
        copyExact("encoder.rnn.bias_ih" + sfx, gatesToTorch(B[d].slice(0, 0, 4 * HIDDEN)));
        copyExact("encoder.rnn.bias_hh" + sfx, gatesToTorch(B[d].slice(0, 4 * HIDDEN, 8 * HIDDEN)));
      }
      count += 3;
    }
    return count;
  }

  std::size_t storeWeights(OnnxFile& f, Head& model)
  {
    torch::NoGradGuard ng;
    std::size_t count = 0;
    for (const auto& [tname, oname] : named(model)) { putTensor(f, f.initializer(oname), param(model, tname)); ++count; }
    const Names n = positional(f);
    putTensor(f, f.initializer(n.matmul[0]), param(model, "encoder.mod_nn.weight").t()); ++count;
    putTensor(f, f.initializer(n.matmul[1]), param(model, "encoder.attn.weight").t()); ++count;
    for (int l = 0; l < 2; ++l)
    {
      std::vector<torch::Tensor> W, R, B;
      for (int d = 0; d < 2; ++d)
      {
        const std::string sfx = "_l" + std::to_string(l) + (d ? "_reverse" : "");
        W.push_back(gatesToOnnx(param(model, "encoder.rnn.weight_ih" + sfx)));
        R.push_back(gatesToOnnx(param(model, "encoder.rnn.weight_hh" + sfx)));
        B.push_back(torch::cat({gatesToOnnx(param(model, "encoder.rnn.bias_ih" + sfx)),
                                gatesToOnnx(param(model, "encoder.rnn.bias_hh" + sfx))}, 0));
      }
      putTensor(f, f.initializer(n.lstm_w[l]), torch::stack(W));
      putTensor(f, f.initializer(n.lstm_r[l]), torch::stack(R));
      putTensor(f, f.initializer(n.lstm_b[l]), torch::stack(B));
      count += 3;
    }
    return count;
  }
}
