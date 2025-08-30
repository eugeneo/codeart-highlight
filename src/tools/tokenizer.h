#ifndef CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H
#define CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H

#include <array>
#include <cstddef>
#include <string_view>

#include "uchen/tensor/special_tensors.h"

namespace codeart::highlight {
namespace impl {
void ExtractEmbeddings(std::span<std::span<const float>> output,
                       std::span<const float> parameters,
                       std::span<const uint32_t> input, size_t embedding_width);
}

class Tokenizer {
 public:
  enum class SpecialTokens : uint32_t {
    kFileBegin = 0,
    kFileEnd = 1,
    kUnknown = 2
  };

  static constexpr size_t kSpecialTokenCount = 3;

  static constexpr size_t kVocabSize =
      /* <begin>/<end>/<unknown> */ kSpecialTokenCount + /* byte*/ 256;

  template <size_t MaxLineLen>
  uchen::core::OneHotTensor<MaxLineLen, kVocabSize> tokenize(
      std::string_view input) const {
    uchen::core::OneHotTensor<MaxLineLen, kVocabSize> tensor;
    Tokenize(input, tensor.data());
    return tensor;
  }

 private:
  static void Tokenize(std::string_view input, std::span<uint32_t> one_hot);
};

template <typename HyperParams>
class EmbeddingsLayer {
 public:
  class Parameters {
   public:
    std::span<const float> data() const { return weights; }

   private:
    std::array<float,
               HyperParams::kEmbeddingDimensions * HyperParams::kTokenTypes>
        weights;
  };

  void set_parameters(const Parameters* params) { params_ = params; }

  template <size_t BatchSize>
  uchen::core::RowProjectionsTensor<BatchSize, HyperParams::kMaxLineLen,
                                    HyperParams::kEmbeddingDimensions>
  operator()(
      const uchen::core::OneHotTensor<BatchSize, HyperParams::kMaxLineLen,
                                      HyperParams::kTokenTypes>& input) const {
    uchen::core::RowProjectionsTensor<BatchSize, HyperParams::kMaxLineLen,
                                      HyperParams::kEmbeddingDimensions>
        result;
    impl::ExtractEmbeddings(result.flat_data(), params_->data(), input.data(),
                            HyperParams::kEmbeddingDimensions);
    return result;
  }

 private:
  const Parameters* params_ = nullptr;
};

}  // namespace codeart::highlight

#endif  // CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H