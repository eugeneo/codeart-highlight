#ifndef CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H
#define CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H

#include <array>
#include <cstddef>
#include <string_view>

#include "uchen/tensor/float_tensor.h"
#include "uchen/tensor/one_hot_tensor.h"

namespace codeart::highlight {

class Tokenizer {
 public:
  enum class SpecialTokens : uint32_t { kBegin = 0, kEnd = 1, kUnknown = 2 };

  static constexpr size_t kVocabSize =
      /* <begin>/<end>/<unknown> */ 3 + /* byte*/ 256;

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

template <size_t EmbeddingDimensions, size_t TokenTypes>
class EmbeddingsLayer {
 public:
  class Parameters {
   public:
   private:
    std::array<float, EmbeddingDimensions * TokenTypes> weights;
  };

  template <size_t BatchSize, size_t MaxLineLen>
  uchen::core::FloatTensor<BatchSize, MaxLineLen, EmbeddingDimensions>
  operator()(const uchen::core::OneHotTensor<BatchSize, MaxLineLen, TokenTypes>&
                 input) const {
    // Embedding logic...
    return {};
  }
};

}  // namespace codeart::highlight

#endif  // CODEART_HIGHLIGHT_SRC_TOOLS_TOKENIZER_H