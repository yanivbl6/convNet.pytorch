#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride, int chunk_size, int bit_mask);

torch::Tensor conv2d_cuda_weight(
    torch::Tensor grad_in,
    torch::Tensor input,
    int kernel_size,
    int chunk_size, int bit_mask);

torch::Tensor conv2d_cuda_input(
    torch::Tensor grad_in,
    torch::Tensor weights,
    int stride,
    int chunk_size, int bit_mask);



// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride, int chunk_size, int bit_mask){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return conv2d_cuda_forward(input, weights, stride, chunk_size, bit_mask);
}

torch::Tensor conv2d_weight(
    torch::Tensor grad_in,
    torch::Tensor input,
    int kernel_size,
    int chunk_size, int bit_mask) {
  
  CHECK_INPUT(grad_in);
  CHECK_INPUT(input);

  return conv2d_cuda_weight(
      grad_in,
      input,
      kernel_size,
      chunk_size, bit_mask);
}

torch::Tensor conv2d_input(
    torch::Tensor grad_in,
    torch::Tensor weights,
    int stride,
    int chunk_size, int bit_mask) {
  
  CHECK_INPUT(grad_in);
  CHECK_INPUT(weights);

  return conv2d_cuda_input(
      grad_in,
      weights,
      stride,
      chunk_size, bit_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv2d_forward, "conv2d forward (CUDA)");
  m.def("weight_grad", &conv2d_weight, "conv2d backward for weights (CUDA)");
  m.def("input_grad", &conv2d_input, "conv2d backward for input (CUDA)");
}