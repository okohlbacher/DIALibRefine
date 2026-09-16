// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause

/// Load a stock peptdeep ONNX file's weights into the libtorch model, and write
/// trained weights back into the SAME file bytes.
///
/// No ONNX or protobuf library. The file is walked as protobuf wire format --
/// ModelProto.graph (7) -> GraphProto.node (1) / initializer (5) -> TensorProto
/// name (8), dims (1), data_type (2), raw_data (9) -- which is enough to find
/// every initializer's bytes and every node's inputs. Write-back overwrites
/// raw_data IN PLACE, same length, graph untouched; verified byte-identical to
/// onnx.save() output before this was written.
///
/// Mapping, by graph POSITION where the exporter renamed things and by NAME
/// where it did not (13 name-matched, 6 LSTM, 2 MatMul = all 21 initializers):
///   LSTM node l, inputs 1,2,3 = W (2,4H,in), R (2,4H,H), B (2,8H)
///     torch rows [i,f,g,o] -> ONNX rows [i,o,f,c]:  onnx = cat(blk0, blk3, blk1, blk2)
///     inverse                                        torch = cat(blk0, blk2, blk3, blk1)
///     Naive stacking matches block 0 only and is wrong by up to 4.57 in block 1.
///   MatMul constants, in graph order: [0] = mod_nn weight^T (103,2), [1] = attn weight^T (256,1)
///   Everything else keeps its torch name (conv weights/biases, rnn_h0/c0, decoder, PReLU slope).
#pragma once

#include <odia/tune/PeptDeepModel.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ODIA::tune
{
  struct OnnxFile
  {
    struct Tensor
    {
      std::string name;
      std::vector<std::int64_t> dims;
      std::int32_t data_type = 0;    ///< 1 = FLOAT
      std::size_t raw_offset = 0;    ///< of raw_data in `bytes`
      std::size_t raw_length = 0;
    };
    struct Node
    {
      std::string op_type;
      std::vector<std::string> inputs;
    };
    std::vector<std::uint8_t> bytes;
    std::vector<Tensor> initializers;
    std::vector<Node> nodes;
    std::vector<std::string> graph_inputs;

    static OnnxFile read(const std::string& path);
    void write(const std::string& path) const;
    const Tensor& initializer(const std::string& name) const;
    bool hasInitializer(const std::string& name) const;
  };

  /// Copy the ONNX weights into @p model. Throws on any shape mismatch or missing tensor.
  /// Returns the number of tensors loaded (21 for both heads).
  std::size_t loadWeights(const OnnxFile& file, Head& model);

  /// Overwrite the ONNX initializers with @p model's weights, in place, in @p file.
  /// Returns the number of tensors written.
  std::size_t storeWeights(OnnxFile& file, Head& model);
}
