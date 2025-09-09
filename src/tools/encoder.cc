#include "src/tools/encoder.h"

#include <cstddef>

#include "absl/log/log.h"  // IWYU pragma: keep

#include "uchen/tensor/function.h"

namespace uchen::layers::details {

using uchen::core::AssignableProjection;
using uchen::core::AssignableTensor;
using uchen::core::BasicTensor;
using uchen::core::TensorProjection;

namespace {

class SoftmaxOperation final
    : public uchen::core::AssignableTensor::Assignable {};

SoftmaxOperation Softmax(
    const uchen::core::AssignableTensor::Assignable& input) {
  return {};
}

class DivideByScalar final : public uchen::core::AssignableTensor::Assignable {
 public:
  constexpr explicit DivideByScalar(float scalar) {}  // : scalar_(scalar) {}

 private:
  // float scalar_;
};

DivideByScalar operator/(const uchen::core::AssignableTensor::Assignable& a,
                         float b) {
  return DivideByScalar(b);
}

}  // namespace

void MultiHeadAttentionImpl::process(const BasicTensor& input,
                                     AssignableTensor& output,
                                     const Parameters& params,
                                     Scratch& scratch) const {
  // const size_t batch_size = input.dim(0);
  // const size_t seq_len = input.dim(1);
  const size_t model_dim = output.dim(2);
  AssignableTensor& kqv = scratch.kqv;
  kqv = input * params.kqv_weights();
  auto k = TensorProjection::dim_slice(kqv, 1, 0, model_dim);
  auto q = TensorProjection::dim_slice(kqv, 1, model_dim, model_dim);
  auto v = TensorProjection::dim_slice(kqv, 1, model_dim * 2, model_dim);
  const size_t HEADS = heads();
  const size_t head_dim = model_dim / HEADS;
  AssignableTensor& smax = scratch.smax;
  for (size_t head = 0; head < HEADS; ++head) {
    auto head_k = TensorProjection::dim_slice(k, 2, head * head_dim, head_dim);
    auto head_q = TensorProjection::dim_slice(q, 2, head * head_dim, head_dim);
    auto head_v = TensorProjection::dim_slice(v, 2, head * head_dim, head_dim);
    smax = Softmax((head_q * uchen::core::Transpose(head_k)) / sqrt(head_dim));
    AssignableProjection::dim_slice(output, 2, head * head_dim, head_dim) =
        smax * head_v;
  }
}

}  // namespace uchen::layers::details