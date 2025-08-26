#include "src/tools/tokenizer.h"

#include <cstdint>

namespace codeart::highlight {

void Tokenizer::Tokenize(std::string_view input, std::span<uint32_t> one_hot) {
  DCHECK_LT(input.length(), one_hot.size() - 2);
  size_t pos = 0;
  std::fill(one_hot.begin(), one_hot.end(),
            static_cast<uint32_t>(SpecialTokens::kUnknown));
  one_hot[pos++] = static_cast<uint32_t>(SpecialTokens::kBegin);
  // TODO: Consider if Highway needs to be involved. I think autovectorizer will
  // do just fine.
  for (char c : input) {
    one_hot[pos++] = static_cast<uint32_t>(c) + 3;
  }
  one_hot[pos++] = static_cast<uint32_t>(SpecialTokens::kEnd);
}

}  // namespace codeart::highlight
