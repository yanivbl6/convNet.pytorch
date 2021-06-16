#include <vector>
#include <torch/extension.h>



template <typename scalar_t>
__global__ void conv2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    scalar_t* __restrict__ output,   //    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    int stride, int chunk_size, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {


    // const int n = blockIdx.y;
    // column index
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("i = %d, b = %d\n", i, b);


    const int c_out = i % channel_out;
    i = i / channel_out;
    const int h = (i % (in_h/stride)) * stride;
    i = i / (in_h/stride);
    const int w = (i % (in_w/stride)) * stride;
    i = i / (in_w/stride);

    const int  chunk_id = i;


    int max_channel_to_calc = (chunk_id+1)*chunk_size;
    if (max_channel_to_calc > channel_in){
        max_channel_to_calc = channel_in;
    }

    float accumulator = 0;
    int* paccumulator = (int*) &accumulator ;


    const int K = (kernel_size-1)/2;


    // if (b == 0 && chunk_id*chunk_size < max_channel_to_calc)
    //     printf("Cout = %d, h = %d, w = %d, b = %d, chunk_id = %d\n", Cout, h, w, b, chunk_id);


    for (int c = chunk_id*chunk_size ; c < max_channel_to_calc ; ++c){
        for (int k_x = 0; k_x < kernel_size; ++k_x){
            const int w_x = h-K+k_x;
            if (w_x >= 0 && w_x < in_h){
                for (int k_y = 0; k_y < kernel_size; ++k_y){
                    const int w_y = w -K + k_y;
                    if (w_y >= 0 && w_y < in_w){
                        accumulator = accumulator + input[b][c][w_x][w_y] * weights[c_out][c][k_x][k_y];
                        (*paccumulator) = (*paccumulator) & bit_mask;
                    }
                }
            }
        }
    }
    const int out_idx =  ((( (b)  *channel_out + c_out)*(in_h/stride)  + h/stride)*(in_w/stride) +  (w/stride));

    //printf("idx= %d, sum = %.02f",out_idx , accumulator );
    atomicAdd( (output+ out_idx ) , accumulator);
}

torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride, int chunk_size, int bit_mask=-1) {

  const auto batch_size = input.size(0);
  const auto in_h = input.size(2);
  const auto in_w = input.size(3);

  const auto channel_in = weights.size(1);
  const auto channel_out = weights.size(0);
  const auto kernel_size = weights.size(2);

  
  auto output = torch::zeros({batch_size,channel_out, in_h/stride, in_w/stride}, input.device());

  const int threads = 1024;

  if (chunk_size <= 0 or chunk_size > channel_in){
      chunk_size = channel_in;
  }

  const int targets = channel_out * (in_h / stride) * (in_w / stride) * ((channel_in + chunk_size - 1) / chunk_size);

  const dim3 blocks((targets + threads - 1) / threads, batch_size);


  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
    conv2d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.data<scalar_t>(), //output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        stride, chunk_size, bit_mask,
        batch_size,
        channel_in,
        channel_out,
        in_h,
        in_w,
        kernel_size);
  }));

  return {output};
}

template <typename scalar_t>
__global__ void conv2d_cuda_backward_kernel_filter(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    scalar_t* __restrict__ grad_weights,   //torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_weights,
    int stride, int chunk_size, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_in = blockIdx.y * blockDim.y + threadIdx.y;

    const int c_out = i % channel_out;
    i = i / channel_out;
    const int k_x = (i % (kernel_size));
    i = i / (kernel_size);
    const int k_y = (i % (kernel_size));
    i = i / (kernel_size);

    const int b = (i % (batch_size));
    i = i / batch_size;

    const int  chunk_id = i;

    const int grad_w = in_w/stride;
    const int grad_h = in_h/stride;

    int max_lines_to_calc = (chunk_id+1)*chunk_size;
    if (max_lines_to_calc > grad_h ){
        max_lines_to_calc = grad_h;
    }

    float accumulator = 0;
    int* paccumulator = (int*) &accumulator ;

    const int K = (kernel_size-1)/2;

    for (int x = 0; x < grad_w; ++x){
        const int in_x = x*stride -K+ k_x;
        if (in_x >= 0 && in_x < in_w){
            for (int y = chunk_id*chunk_size; y < max_lines_to_calc; ++y){
                const int in_y = y*stride -K + k_y;
                if (in_y >= 0 && in_y < in_h){
                    accumulator = accumulator + (grad_in[b][c_out][y][x] * input[b][c_in][in_y][in_x]);
                    (*paccumulator) = (*paccumulator) & bit_mask;
                }
            }
        }
    }
    const int out_idx =  ((( (c_out)  *channel_in + c_in)*kernel_size  + k_y)*kernel_size +  (k_x));

    atomicAdd( (grad_weights+ out_idx ) , accumulator);
    
}


torch::Tensor conv2d_cuda_weight(
    torch::Tensor grad_in,
    torch::Tensor input,
    int kernel_size,
    int chunk_size, int bit_mask=-1) {



  const auto batch_size = input.size(0);
  const auto channels = input.size(1);
  const auto in_h = input.size(2);
  const auto in_w = input.size(3);

  const auto channel_in = input.size(1);
  const auto channel_out = grad_in.size(1);

  const auto grad_h = grad_in.size(2);
  const auto grad_w = grad_in.size(3);

  const int stride =  in_h/grad_h;

  //auto grad_weights = torch::zeros_like(weights);
  auto grad_weights = torch::zeros({channel_out,channel_in, kernel_size , kernel_size }, input.device());
  const int threads = 1024;

  if (chunk_size <= 0 or chunk_size > grad_h){
      chunk_size = grad_h;
  }

  const int targets = channel_out *  kernel_size * kernel_size * batch_size * ((grad_h  + chunk_size - 1) / chunk_size);

  const dim3 blocks((targets + threads - 1) / threads, channel_in);

  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "conv2d_cuda_backward_kernel_filter", ([&] {
    conv2d_cuda_backward_kernel_filter<scalar_t><<<blocks, threads>>>(
        grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_weights.data<scalar_t>(),  // grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        stride, chunk_size, bit_mask,
        batch_size,
        channel_in,
        channel_out,
        in_h,
        in_w,
        kernel_size);
  }));

  return grad_weights;
}

template <typename scalar_t>
__global__ void conv2d_cuda_backward_kernel_input(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    scalar_t* __restrict__ grad_out,   //torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out,
    int stride, int chunk_size, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;

    const int c_in = i % channel_in;
    i = i / channel_in;
    const int h = (i % in_h);
    i = i / in_h;
    const int w = (i % in_w);
    i = i / in_w;

    const int  chunk_id = i;

    int max_channel_to_calc = (chunk_id+1)*chunk_size;
    if (max_channel_to_calc > channel_out){
        max_channel_to_calc = channel_out;
    }

    float accumulator = 0;
    int* paccumulator = (int*) &accumulator ;

    const int K = (kernel_size-1)/2;

    for (int c_out = chunk_id*chunk_size ; c_out < max_channel_to_calc ; ++c_out){
        for (int k_y = 0; k_y < kernel_size; ++k_y){
            int g_y = h + K - k_y;
            if (g_y >= 0 && g_y < in_h && g_y % stride == 0 ){
                g_y = g_y / stride;
                for (int k_x = 0; k_x < kernel_size; ++k_x){
                    int g_x = w + K - k_x;
                    if (g_x >= 0 && g_x < in_w && g_x % stride == 0){
                        g_x = g_x / stride;
                        accumulator = accumulator + grad_in[b][c_out][g_y][g_x] * weights[c_out][c_in][ k_y][k_x];
                        (*paccumulator) = (*paccumulator) & bit_mask;
                    }
                }
            }
        }
    }
    const int out_idx =  ((( (b)  *channel_in + c_in)*(in_h)  + h)*(in_w) +  (w));

    atomicAdd( (grad_out+ out_idx ) , accumulator);
    
}


torch::Tensor conv2d_cuda_input(
    torch::Tensor grad_in,
    torch::Tensor weights,
    int stride,
    int chunk_size, int bit_mask=-1){

  const auto grad_h = grad_in.size(2);
  const auto grad_w = grad_in.size(3);

  const auto batch_size = grad_in.size(0);
  const auto in_h = stride*grad_h;
  const auto in_w = stride*grad_w;

  const auto channel_in = weights.size(1);
  const auto channel_out = weights.size(0);
  const auto kernel_size = weights.size(2);


  auto grad_out = torch::zeros({batch_size,channel_in, in_h , in_w }, grad_in.device());

  const int threads = 1024;

  if (chunk_size <= 0 or chunk_size > channel_out){
      chunk_size = channel_out;
  }
    
  const int targets = channel_in * in_h * in_w * ((channel_out + chunk_size - 1) / chunk_size);


  const dim3 blocks((targets + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "conv2d_cuda_backward_kernel_input", ([&] {
    conv2d_cuda_backward_kernel_input<scalar_t><<<blocks, threads>>>(
        grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_out.data<scalar_t>(),  // grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        stride, chunk_size, bit_mask,
        batch_size,
        channel_in,
        channel_out,
        in_h,
        in_w,
        kernel_size);
  }));


  return grad_out;
}

