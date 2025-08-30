#include "src/tools/tokenizer.h"

#include <cstdint>

namespace codeart::highlight {
namespace impl {

void ExtractEmbeddings(std::span<std::span<const float>> output,
                       std::span<const float> parameters,
                       std::span<const uint32_t> input,
                       size_t embedding_width) {
  DCHECK_EQ(output.size(), input.size());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = parameters.subspan(input[i] * embedding_width, embedding_width);
  }
}

}  // namespace impl

void Tokenizer::Tokenize(std::string_view input, std::span<uint32_t> one_hot) {
  DCHECK_LT(input.length(), one_hot.size() - 2);
  size_t pos = 0;
  std::fill(one_hot.begin(), one_hot.end(),
            static_cast<uint32_t>(SpecialTokens::kUnknown));
  one_hot[pos++] = static_cast<uint32_t>(SpecialTokens::kFileBegin);
  // TODO: Consider if Highway needs to be involved. I expect autovectorizer to
  // do just fine, but may want to hand roll.
  for (char c : input) {
    one_hot[pos++] = static_cast<uint32_t>(c) + kSpecialTokenCount;
  }
  one_hot[pos++] = static_cast<uint32_t>(SpecialTokens::kFileEnd);
}

}  // namespace codeart::highlight
