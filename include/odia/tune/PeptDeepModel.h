// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// AlphaPeptDeep's RT and CCS models, transcribed to libtorch.
///
/// The transcription is peptdeep 1.5.1 building_block.py, verified against
/// torch to 4.5e-8 (RT) / 6.1e-5 (CCS) by an independent numpy re-implementation
/// before this was written. Every layer, in forward order:
///
///   one_hot(aa_indices, 27)                                   (b,L,27)
///   mod_x[..., :6] ++ mod_x[..., 6:] @ Wm^T   Wm: Linear(103->2, no bias)   (b,L,8)
///   [++ charge*0.1 broadcast over L]           CCS only                    35 | 36 ch
///   x ++ Conv1d(k3,p1)(x) ++ Conv1d(k5,p2)(x) ++ Conv1d(k7,p3)(x)   no activation   4C
///   BiLSTM, 2 layers, hidden 128/dir, frozen h0/c0 (4,1,128)                (b,L,256)
///   softmax_L(x @ wa^T) . x     wa: Linear(256->1, no bias)                  (b,256)
///   dropout 0.1 (train only)    [++ charge*0.1]                              256 | 257
///   Linear(->64) -> PReLU(one slope) -> Linear(->1)                          (b,)
///
/// rt_pred is rt_norm (the run's scale is NOT in the model); ccs_pred is CCS
/// in A^2. The Python predict() clamps at 0; ONNX and this do not.
///
/// Parameter names mirror peptdeep's state_dict so a debugger sees the same
/// tree; the ONNX side is loaded by graph POSITION (see OnnxWeights.h).
#pragma once

#include <torch/torch.h>

namespace ODIA::tune
{
  constexpr std::int64_t AA_CLASSES = 27;
  constexpr std::int64_t MOD_FEATURES = 109;
  constexpr std::int64_t MOD_FIX = 6;          ///< first six mod features pass through
  constexpr std::int64_t MOD_EMBED = 2;
  constexpr std::int64_t HIDDEN = 128;
  constexpr float CHARGE_SCALE = 0.1f;          ///< peptdeep charge_factor; the ONNX takes charge*0.1

  struct EncoderImpl : torch::nn::Module
  {
    bool with_charge;
    torch::nn::Linear mod_nn{nullptr};
    torch::nn::Conv1d cnn_short{nullptr}, cnn_medium{nullptr}, cnn_long{nullptr};
    torch::nn::LSTM rnn{nullptr};
    torch::Tensor rnn_h0, rnn_c0;
    torch::nn::Linear attn{nullptr};

    explicit EncoderImpl(bool with_charge_) : with_charge(with_charge_)
    {
      const std::int64_t C = AA_CLASSES + MOD_FIX + MOD_EMBED + (with_charge ? 1 : 0);
      mod_nn = register_module("mod_nn", torch::nn::Linear(torch::nn::LinearOptions(MOD_FEATURES - MOD_FIX, MOD_EMBED).bias(false)));
      cnn_short  = register_module("cnn_short",  torch::nn::Conv1d(torch::nn::Conv1dOptions(C, C, 3).padding(1)));
      cnn_medium = register_module("cnn_medium", torch::nn::Conv1d(torch::nn::Conv1dOptions(C, C, 5).padding(2)));
      cnn_long   = register_module("cnn_long",   torch::nn::Conv1d(torch::nn::Conv1dOptions(C, C, 7).padding(3)));
      rnn = register_module("rnn", torch::nn::LSTM(torch::nn::LSTMOptions(4 * C, HIDDEN).num_layers(2).bidirectional(true).batch_first(true)));
      // Frozen initial states, exactly as peptdeep: parameters with requires_grad = false.
      rnn_h0 = register_parameter("rnn_h0", torch::zeros({4, 1, HIDDEN}), /*requires_grad=*/false);
      rnn_c0 = register_parameter("rnn_c0", torch::zeros({4, 1, HIDDEN}), /*requires_grad=*/false);
      attn = register_module("attn", torch::nn::Linear(torch::nn::LinearOptions(2 * HIDDEN, 1).bias(false)));
    }

    /// @param aa      int64 [b, L]
    /// @param mod_x   float [b, L, 109]
    /// @param charges float [b, 1], already scaled by CHARGE_SCALE; ignored unless with_charge
    torch::Tensor forward(const torch::Tensor& aa, const torch::Tensor& mod_x, const torch::Tensor& charges)
    {
      namespace F = torch::nn::functional;
      const auto b = aa.size(0), L = aa.size(1);
      auto onehot = F::one_hot(aa, AA_CLASSES).to(mod_x.dtype());
      auto mod8 = torch::cat({mod_x.slice(2, 0, MOD_FIX), mod_nn->forward(mod_x.slice(2, MOD_FIX, MOD_FEATURES))}, 2);
      std::vector<torch::Tensor> parts{onehot, mod8};
      if (with_charge) { parts.push_back(charges.unsqueeze(1).expand({b, L, 1})); }
      auto x = torch::cat(parts, 2);                                   // (b,L,C)
      auto xc = x.transpose(1, 2);                                     // (b,C,L)
      x = torch::cat({x, cnn_short->forward(xc).transpose(1, 2),
                         cnn_medium->forward(xc).transpose(1, 2),
                         cnn_long->forward(xc).transpose(1, 2)}, 2);    // (b,L,4C)
      auto h0 = rnn_h0.expand({4, b, HIDDEN}).contiguous();
      auto c0 = rnn_c0.expand({4, b, HIDDEN}).contiguous();
      auto out = std::get<0>(rnn->forward(x, std::make_tuple(h0, c0))); // (b,L,256)
      auto a = torch::softmax(attn->forward(out), 1);                  // (b,L,1)
      return (out * a).sum(1);                                         // (b,256)
    }
  };
  TORCH_MODULE(Encoder);

  /// One head: RT (`with_charge=false`) or CCS (`with_charge=true`).
  struct HeadImpl : torch::nn::Module
  {
    bool ccs;
    Encoder encoder{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear dec0{nullptr}, dec2{nullptr};
    torch::nn::PReLU prelu{nullptr};

    explicit HeadImpl(bool ccs_) : ccs(ccs_)
    {
      encoder = register_module("encoder", Encoder(ccs));
      dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.1)));
      dec0 = register_module("dec0", torch::nn::Linear(2 * HIDDEN + (ccs ? 1 : 0), 64));
      prelu = register_module("prelu", torch::nn::PReLU());            // one shared slope, init 0.25
      dec2 = register_module("dec2", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(const torch::Tensor& aa, const torch::Tensor& mod_x, const torch::Tensor& charges)
    {
      auto v = dropout->forward(encoder->forward(aa, mod_x, charges));
      if (ccs) { v = torch::cat({v, charges}, 1); }
      return dec2->forward(prelu->forward(dec0->forward(v))).squeeze(1);
    }

    /// The parameters an optimizer may touch: everything except the frozen states.
    std::vector<torch::Tensor> trainable()
    {
      std::vector<torch::Tensor> out;
      for (auto& p : parameters()) { if (p.requires_grad()) { out.push_back(p); } }
      return out;
    }
  };
  TORCH_MODULE(Head);
}
