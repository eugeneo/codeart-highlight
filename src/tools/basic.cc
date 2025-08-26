#include <sys/stat.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"

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

template <size_t MaxLineLen, size_t EmbeddingDimensions,
          size_t EncoderDimensions>
class EncoderLayer {
 public:
  template <size_t BatchSize>
  FloatTensor<BatchSize, MaxLineLen, EncoderDimensions> operator()(
      const FloatTensor<BatchSize, MaxLineLen, EmbeddingDimensions>& embeddings)
      const {
    //         X = embeddings
    // for i in 0..N-1:
    //     attn_out  = MultiHeadAttention(X)
    //     X         = ResidualNorm(X, Dropout(attn_out))
    //     ffn_out   = FeedForward(X)
    //     X         = ResidualNorm(X, Dropout(ffn_out))
    // encoded = X
    return {};
  }
};

template <size_t MaxLineLen, size_t EmbeddingDimensions, size_t TokenTypes>
class ClassificationLayer {
 public:
  template <size_t BatchSize>
  FloatTensor<BatchSize, MaxLineLen, TokenTypes> operator()(
      const FloatTensor<BatchSize, MaxLineLen, TokenTypes>& encoded) const {
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

template <typename I, typename L>
class LayerStacking {
 public:
  LayerStacking(const I& input, const L& layer)
      : input_(input), layer_(layer) {}

  auto result() const { return layer_(input_); }

  template <typename L1>
  auto operator>(const L1& next_layer) const {
    return LayerStacking<std::remove_cvref_t<decltype(result())>, L1>(
        result(), next_layer);
  }

 private:
  const I& input_;
  const L& layer_;
};

template <typename T>
class InputRef {
 public:
  explicit InputRef(const T& input) : input_(input) {}

  template <typename L>
  auto operator>(const L& layer) const {
    return LayerStacking(input_, layer);
  }

 private:
  const T& input_;
};

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
  static constexpr size_t kEncodeDims = 256;

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
    return uchen::layers::DecodeCRF<TokenType>(dnn(one_hot));
  }

 private:
  auto dnn(const auto& input) const {
    return std::apply(
        [&](auto&... ls) {
          return (uchen::layers::InputRef(input) > ... > ls).result();
        },
        layers_);
  }

  std::tuple<
      codeart::highlight::EmbeddingsLayer<
          EmbeddingDimensions, codeart::highlight::Tokenizer::kVocabSize>,
      uchen::layers::EncoderLayer<MaxLineLen, EmbeddingDimensions, kEncodeDims>,
      uchen::layers::ClassificationLayer<MaxLineLen, kEncodeDims, 256>>
      layers_;
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
  CodeartHighlightModel<200, 10> model;
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