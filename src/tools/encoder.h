#ifndef SRC_TOOLS_ENCODER_H_
#define SRC_TOOLS_ENCODER_H_

#include <cstddef>

#include "absl/log/check.h"

#include "src/tools/neural_network.h"
#include "uchen/tensor/float_tensor.h"

namespace uchen::layers {

namespace details {
class MultiHeadAttentionImpl {
 public:
  class Parameters {
   public:
    virtual ~Parameters() = default;

    virtual const uchen::core::BasicTensor& kqv_weights() const = 0;
  };

  struct Scratch {
    uchen::core::AssignableTensor& kqv;
    uchen::core::AssignableTensor& smax;
  };

  constexpr explicit MultiHeadAttentionImpl(size_t heads) : heads_(heads) {}

  void process(const uchen::core::BasicTensor& input,
               uchen::core::AssignableTensor& output, const Parameters& params,
               Scratch& scratch) const;

  size_t heads() const { return heads_; }

 private:
  size_t heads_;
};

}  // namespace details

template <size_t SeqLen, size_t Dimensions, size_t Heads>
  requires(Dimensions % Heads == 0)
class MultiHeadAttention : private details::MultiHeadAttentionImpl {
 public:
  class Parameters final : public details::MultiHeadAttentionImpl::Parameters {
   public:
    const uchen::core::BasicTensor& kqv_weights() const override {
      return kqv_weights_;
    }

   private:
    uchen::core::FloatTensor<Dimensions * 3 * Heads, Dimensions> kqv_weights_;
  };

  template <size_t BatchSize>
  using result_t = uchen::core::FloatTensor<BatchSize, SeqLen, Dimensions>;

  constexpr MultiHeadAttention() : details::MultiHeadAttentionImpl(Heads) {}

  constexpr void set_parameters(const Parameters* params) { params_ = params; }

  template <uchen::core::BatchTensorWithDims<SeqLen, Dimensions> Input>
  result_t<Input::kDims[0]> operator()(const Input& input) const {
    CHECK_NE(params_, nullptr);
    result_t<Input::kDims[0]> output;
    core::FloatTensor<Input::kDims[0], SeqLen * 3, Dimensions> kqv_scratch;
    core::FloatTensor<Input::kDims[0], SeqLen, SeqLen / Heads> smax_scratch_;
    details::MultiHeadAttentionImpl::Scratch kqv_scratch_struct{
        .kqv = kqv_scratch, .smax = smax_scratch_};
    process(input, output, *params_, kqv_scratch_struct);
    return output;
  }

 private:
  const Parameters* params_ = nullptr;
};

template <size_t SeqLen, size_t Dimensions, size_t AttentionHeads>
  requires(SeqLen > 0 && Dimensions > 0 && AttentionHeads > 0 &&
           Dimensions % AttentionHeads == 0)
class EncoderLayer {
 public:
  struct Parameters {
    MultiHeadAttention<SeqLen, Dimensions, AttentionHeads>::Parameters
        attention;
  };

  constexpr void set_parameters(const Parameters* params) {
    dnn_.template get_layer<0>().set_parameters(&params->attention);
  }

  template <uchen::core::BatchTensorWithDims<SeqLen, Dimensions> Input>
  uchen::core::FloatTensor<Input::kDims[0], SeqLen, Dimensions> operator()(
      const Input& embeddings) const {
    return dnn_(embeddings);
  }

 private:
  uchen::core::NeuralNetwork<
      // TODO: Repeat for number of blocks
      MultiHeadAttention<SeqLen, Dimensions, AttentionHeads>
      // TODO:
      //     X         = ResidualNorm(X, Dropout(attn_out))
      //     ffn_out   = FeedForward(X)
      //     X         = ResidualNorm(X, Dropout(ffn_out))
      >
      dnn_;
};

}  // namespace uchen::layers

#endif  // SRC_TOOLS_ENCODER_H_
