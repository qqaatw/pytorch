#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

Tensor make_per_channel_quantized_tensor_cuda(
  const Tensor& self,
  const Tensor& scales,
  const Tensor& zero_points,
  int64_t axis) {
    Tensor dst = at::_empty_per_channel_affine_quantized(
    self.sizes(),
    scales,
    zero_points,
    axis,
    self.options().dtype(toQIntType(self.scalar_type())));
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_channel_quantized_tensor_cuda", [&]() {
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(dst)
          .add_input(self)
          .build();
        gpu_kernel(iter, [] GPU_LAMBDA(underlying_t value) -> scalar_t {
          return scalar_t(value);
        });
      });
  return dst;
}

} // namespace native
} // namespace at
