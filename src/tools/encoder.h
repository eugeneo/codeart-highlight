#ifndef SRC_TOOLS_ENCODER_H_
#define SRC_TOOLS_ENCODER_H_

#include <cstddef>

#include "src/tools/neural_network.h"
#include "uchen/tensor/float_tensor.h"

namespace uchen::layers {

template <size_t Tokens, size_t EncoderDimensions, size_t Heads>
  requires(EncoderDimensions % Heads == 0)
class MultiHeadAttention {
 public:
  template <uchen::core::TensorLike Input>
    requires(Input::kDims.size() == 3 && Input::kDims[1] == Tokens &&
             Input::kDims[2] == EncoderDimensions)
  uchen::core::FloatTensor<Input::kDims[0], Tokens, EncoderDimensions>
  operator()(const uchen::core::TensorLike auto& input) const {
    // Apply multi-head attention mechanism
    return {};
  }
};

template <size_t MaxLineLen, size_t EncoderDimensions, size_t AttentionHeads>
class EncoderLayer {
 public:
  template <uchen::core::TensorLike Input>
    requires(Input::kDims.size() == 3 && Input::kDims[1] == MaxLineLen &&
             Input::kDims[2] == EncoderDimensions)
  uchen::core::FloatTensor<Input::kDims[0], MaxLineLen, EncoderDimensions>
  operator()(const Input& embeddings) const {
    //         X = embeddings
    // for i in 0..N-1:
    //     attn_out  = MultiHeadAttention(X)
    //     X         = ResidualNorm(X, Dropout(attn_out))
    //     ffn_out   = FeedForward(X)
    //     X         = ResidualNorm(X, Dropout(ffn_out))
    // encoded = X
    return {};
  }

 private:
  uchen::core::NeuralNetwork<
      MultiHeadAttention<MaxLineLen, EncoderDimensions, AttentionHeads>>
      dnn_;
};

}  // namespace uchen::layers

#endif  // SRC_TOOLS_ENCODER_H_
