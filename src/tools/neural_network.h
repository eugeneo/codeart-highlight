#ifndef SRC_TOOLS_NEURAL_NETWORK_H_
#define SRC_TOOLS_NEURAL_NETWORK_H_

#include <tuple>
namespace uchen::core {

template <typename... Layers>
  requires(sizeof...(Layers) > 0)
class NeuralNetwork {


 public:
  constexpr NeuralNetwork() = default;

  template <size_t I>
    requires(I < sizeof...(Layers))
  auto& get_layer() {
    return std::get<I>(layers_);
  }

  auto operator()(const auto& input) const {
    return std::apply(
        [&](auto&... ls) { return (InputRef(input) > ... > ls).result(); },
        layers_);
  }

 private:
  template <typename T>
  class InputRef {
   public:
    explicit constexpr InputRef(const T& input) : input_(input) {}

    template <typename L>
    constexpr auto operator>(const L& layer) const {
      return LayerStacking(input_, layer);
    }

   private:
    const T& input_;
  };

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

  std::tuple<Layers...> layers_;
};

}  // namespace uchen::core

#endif  // SRC_TOOLS_NEURAL_NETWORK_H_