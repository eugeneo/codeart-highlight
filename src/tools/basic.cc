#include <sys/stat.h>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"

#include "src/tools/encoder.h"
#include "src/tools/neural_network.h"
#include "src/tools/tokenizer.h"
#include "uchen/tensor/float_tensor.h"

namespace uchen::layers {

using uchen::core::FloatTensor;

template <typename T>
struct BIOToken {
  enum class BIO { kBegin, kInside, kOutside };
  T value;
  BIO bio;
};

template <size_t MaxLineLen, size_t EmbeddingDimensions, size_t TokenTypes>
class ClassificationLayer {
 public:
  template <size_t BatchSize>
  FloatTensor<BatchSize, MaxLineLen, TokenTypes> operator()(
      const FloatTensor<BatchSize, MaxLineLen, EmbeddingDimensions>& encoded)
      const {
    return {};
  }
};

template <typename TokenType>
auto DecodeCRF(const auto& logits) {
  std::array<std::array<BIOToken<TokenType>,
                        std::remove_cvref_t<decltype(logits)>::dims[1]>,
             std::remove_cvref_t<decltype(logits)>::dims[0]>
      result;
  for (auto& line : result) {
    line.fill(BIOToken<TokenType>{TokenType::kUnknown,
                                  BIOToken<TokenType>::BIO::kOutside});
  }
  return result;
}

}  // namespace uchen::layers

namespace {
enum class TokenType {
  kKeyword,
  kIdentifier,
  kLiteral,
  kOperator,
  kPunctuation,
  kComment,
  kWhitespace,
  kUnknown
};

template <size_t MaxLineLen, size_t EmbeddingDimensions>
class CodeartHighlightModel {
 public:
  constexpr CodeartHighlightModel() {
    dnn_.template get_layer<0>().set_parameters(&embeddings_params_);
  }

  template <size_t BatchSize>
  std::array<std::array<uchen::layers::BIOToken<TokenType>, MaxLineLen>,
             BatchSize>
  Highlight(std::span<const std::string_view, BatchSize> code) const {
    std::array<std::array<uchen::layers::BIOToken<TokenType>, MaxLineLen>,
               BatchSize>
        result;
    for (auto& line : result) {
      line.fill(uchen::layers::BIOToken<TokenType>{
          TokenType::kUnknown,
          uchen::layers::BIOToken<TokenType>::BIO::kOutside});
    }
    codeart::highlight::Tokenizer tokenizer;
    uchen::core::OneHotTensor<BatchSize, MaxLineLen, tokenizer.kVocabSize>
        one_hot;
    for (size_t i = 0; i < BatchSize; ++i) {
      one_hot[i] = tokenizer.tokenize<MaxLineLen>(code[i]);
    }
    return uchen::layers::DecodeCRF<TokenType>(dnn_(one_hot));
  }

 private:
  struct HyperParams {
    static constexpr size_t kMaxLineLen = MaxLineLen;
    static constexpr size_t kEmbeddingDimensions = EmbeddingDimensions;
    static constexpr size_t kTokenTypes =
        codeart::highlight::Tokenizer::kVocabSize;
  };

  codeart::highlight::EmbeddingsLayer<HyperParams>::Parameters
      embeddings_params_;

  uchen::core::NeuralNetwork<
      codeart::highlight::EmbeddingsLayer<HyperParams>,
      uchen::layers::EncoderLayer<MaxLineLen, EmbeddingDimensions, 4>,
      uchen::layers::ClassificationLayer<MaxLineLen, EmbeddingDimensions, 256>>
      dnn_;
};

char TokenLabel(const uchen::layers::BIOToken<TokenType>& type) {
  switch (type.value) {
    case TokenType::kKeyword:
      return 'K';
    case TokenType::kIdentifier:
      return 'I';
    case TokenType::kLiteral:
      return 'L';
    case TokenType::kOperator:
      return 'O';
    case TokenType::kPunctuation:
      return 'P';
    case TokenType::kComment:
      return 'C';
    case TokenType::kWhitespace:
      return 'W';
    case TokenType::kUnknown:
      return 'u';
    default:
      return '?';
  }
}

void ClassifyTokens(std::string_view code) {
  CodeartHighlightModel<200, 16> model;
  LOG(INFO) << "Model size: " << sizeof(model);
  std::span<const std::string_view, 1> batch(&code, 1);
  auto tokens = model.Highlight(batch);
  LOG(INFO) << "Code: " << code;
  LOG(INFO) << "Tokens: "
            << absl::StrJoin(tokens[0], "", [](auto* out, const auto& t) {
                 (*out) += TokenLabel(t);
               });
}

}  // namespace

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  ClassifyTokens("int a = 2;");
  ClassifyTokens("int main() {\n  return 0;\n}");
  return 0;
}