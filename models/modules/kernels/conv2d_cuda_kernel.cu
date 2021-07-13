#include <vector>
#include <torch/extension.h>
#include <assert.h>

#include <math.h>
template <typename scalar_t>
__global__ void conv2d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    scalar_t* __restrict__ output,   //    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    int stride, int chunk_size, int chunks, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {

    __shared__ scalar_t reduction_vec[1024];

    // const int n = blockIdx.y;
    // column index
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;

    const int b = blockIdx.y;
    const int chunk_id = threadIdx.y;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int c_out = i % channel_out;
    i = i / channel_out;
    const int h = (i % (in_h/stride)) * stride;
    i = i / (in_h/stride);
    const int w = (i % (in_w/stride)) * stride;
    i = i / (in_w/stride);
    //printf("%d %d %d %d %d \n" , chunk_id,  b, c_out, h, w );
    if (!i){
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
                const int w_x = w-K+k_x;
                if (w_x >= 0 && w_x < in_w){
                    for (int k_y = 0; k_y < kernel_size; ++k_y){
                        const int w_y = h -K + k_y;
                        if (w_y >= 0 && w_y < in_h){
                            accumulator = accumulator + input[b][c][w_y][w_x] * weights[c_out][c][k_y][k_x];
                            (*paccumulator) = (*paccumulator) & bit_mask;
                        }
                    }
                }
            }
        }
        const int out_idx =  ((( (b)  *channel_out + c_out)*(in_h/stride)  + h/stride)*(in_w/stride) +  (w/stride));
        
        if (chunks > 1){

            reduction_vec[ threadIdx.x*chunks + (threadIdx.y)] = accumulator;

            //int chunks_left = (chunks >> 1) << 1;

            // while (chunks_left > 0){
            //     __syncthreads();
            //     if (threadIdx.y > chunks_left){
            //         reduction_vec[ threadIdx.x*chunks + (threadIdx.y-chunks_left)] += reduction_vec[ threadIdx.x*chunks + (threadIdx.y)];
            //         *(preduction_vec  + threadIdx.x*chunks + (threadIdx.y-chunks_left))  =  *(preduction_vec  + threadIdx.x*chunks + (threadIdx.y-chunks_left)) & bit_mask;
            //     }
            //     chunks_left = chunks_left >> 1;
            // }
            __syncthreads();

            // if (threadIdx.y == 0){
            //     output[out_idx] = preduction_vec[threadIdx.x  * chunks];
            // }

            if (threadIdx.y == 0){
                for (int i = chunks-1; i > 0; --i){
                    accumulator += reduction_vec[ threadIdx.x*chunks + i];
                    (*paccumulator) = (*paccumulator) & bit_mask;
                    //*(preduction_vec  + threadIdx.x*chunks + (threadIdx.y-chunks_left))  =  *(preduction_vec  + threadIdx.x*chunks + (threadIdx.y-chunks_left)) & bit_mask;
                }
                output[out_idx] = accumulator;
                //output[out_idx] = preduction_vec[threadIdx.x  * chunks];
            }
        } else{
            output[out_idx] = accumulator;
        }
        

        //printf("idx= %d, sum = %.02f",out_idx , accumulator );
        //atomicAdd( (output+ out_idx ) , accumulator);
    }
}

torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int stride, int chunks_count, int bit_mask=-1) {



  const auto batch_size = input.size(0);
  const auto in_h = input.size(2);
  const auto in_w = input.size(3);

  const auto channel_in = weights.size(1);
  const auto channel_out = weights.size(0);
  const auto kernel_size = weights.size(2);

  if (chunks_count == 0){
      chunks_count = int(sqrt(kernel_size*kernel_size*channel_in));
      if (chunks_count > 1024){
          chunks_count = 1024;
      }
  }

  const int chunk_size = (channel_in + chunks_count - 1) / chunks_count;

  auto output = torch::zeros({batch_size,channel_out, in_h/stride, in_w/stride}, input.device());


  const int chunks = (channel_in + chunk_size - 1) / chunk_size;
  assert(chunks <= 1024);

  const int threads_x = (1024 / chunks);

  const dim3 threads(threads_x , chunks);


  const int targets = channel_out * (in_h / stride) * (in_w / stride); 

  const dim3 blocks((targets + threads_x - 1) / (threads_x), batch_size);


  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_forward_cuda", ([&] {
    conv2d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.data<scalar_t>(), //output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        stride, chunk_size, chunks, bit_mask,
        batch_size,
        channel_in,
        channel_out,
        in_h,
        in_w,
        kernel_size);
  }));

  return output;
}

template <typename scalar_t>
__global__ void conv2d_cuda_backward_kernel_filter(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    scalar_t* __restrict__ grad_weights,   //torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_weights,
    int stride, int chunk_size, int chunks, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {

    __shared__ scalar_t reduction_vec[1024];
    const int c_in = blockIdx.y;
    int chunk_id = threadIdx.y;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int c_out = i % channel_out;
    i = i / channel_out;
    const int k_x = (i % (kernel_size));
    i = i / (kernel_size);
    const int k_y = (i % (kernel_size));
    i = i / (kernel_size);



    if (!i){


        const int grad_w = in_w/stride;
        const int grad_h = in_h/stride;

        float accumulator = 0;
        int* paccumulator = (int*) &accumulator ;
        const int K = (kernel_size-1)/2;


        if (chunks % batch_size){
            int max_elements_to_calc = (chunk_id+1)*chunk_size;
            if (max_elements_to_calc > grad_h*grad_w*batch_size ){
                max_elements_to_calc = grad_h*grad_w*batch_size ;
            }

            for (int e = chunk_id*chunk_size ; e < max_elements_to_calc; ++e){
                int ep = e;
                const int b =  ep % batch_size;
                ep = ep / batch_size;
                const int y =  ep % grad_h;
                const int in_y = y*stride -K + k_y;

                if (in_y >= 0 && in_y < in_h){
                    ep = ep / grad_h;
                    const int x = ep % grad_w;
                    const int in_x = x*stride -K+ k_x;
                    if (in_x >= 0 && in_x < in_w){
                        accumulator = accumulator + (grad_in[b][c_out][y][x] * input[b][c_in][in_y][in_x]);
                        (*paccumulator) = (*paccumulator) & bit_mask;
                    }
                }
            }
        } else if (chunks % (batch_size*grad_h)){
            const int b = chunk_id % batch_size;
            chunk_id = chunk_id / batch_size;

            int max_elements_to_calc = (chunk_id+1)*(chunk_size);
            if (max_elements_to_calc > grad_h*grad_w ){
                max_elements_to_calc = grad_h*grad_w;
            }

            for (int e = chunk_id*chunk_size ; e < max_elements_to_calc; ++e){
                int ep = e;
                const int y =  ep % grad_h;
                const int in_y = y*stride -K + k_y;

                if (in_y >= 0 && in_y < in_h){
                    ep = ep / grad_h;
                    const int x = ep % grad_w;
                    const int in_x = x*stride -K+ k_x;

                    if (in_x >= 0 && in_x < in_w){
                        accumulator = accumulator + (grad_in[b][c_out][y][x] * input[b][c_in][in_y][in_x]);
                        (*paccumulator) = (*paccumulator) & bit_mask;
                    }
                }
            }
        } else{
            const int b = chunk_id % batch_size;
            chunk_id = chunk_id/ batch_size;
            const int y = chunk_id % grad_h;
            chunk_id = chunk_id/ grad_h;
            const int in_y = y*stride -K + k_y;

            if (in_y >= 0 && in_y < in_h){

                int max_elements_to_calc = (chunk_id+1)*(chunk_size);
                if (max_elements_to_calc > grad_w ){
                    max_elements_to_calc = grad_w;
                }

                for (int x = chunk_id*chunk_size ; x < max_elements_to_calc; ++x){
                    const int in_x = x*stride -K+ k_x;
                    if (in_x >= 0 && in_x < in_w){
                        accumulator = accumulator + (grad_in[b][c_out][y][x] * input[b][c_in][in_y][in_x]);
                        (*paccumulator) = (*paccumulator) & bit_mask;
                    }
                }
            }
        }

        const int out_idx =  ((( (c_out)  *channel_in + c_in)*kernel_size  + k_y)*kernel_size +  (k_x));

        if (chunks > 1){
            reduction_vec[ threadIdx.x*chunks + (threadIdx.y)] = accumulator;
            __syncthreads();
            if (threadIdx.y == 0){
                for (int i = chunks-1; i > 0; --i){
                    accumulator += reduction_vec[ threadIdx.x*chunks + i];
                    (*paccumulator) = (*paccumulator) & bit_mask;
                }
                grad_weights[out_idx] = accumulator;
            }
        } else{
            grad_weights[out_idx] = accumulator;
        }

        //atomicAdd( (grad_weights+ out_idx ) , accumulator);
    }
}


torch::Tensor conv2d_cuda_weight(
    torch::Tensor grad_in,
    torch::Tensor input,
    int kernel_size, int chunks_count, int bit_mask=-1) {



  const auto batch_size = input.size(0);
  const auto channels = input.size(1);
  const auto in_h = input.size(2);
  const auto in_w = input.size(3);

  const auto channel_in = input.size(1);
  const auto channel_out = grad_in.size(1);

  const auto grad_h = grad_in.size(2);
  const auto grad_w = grad_in.size(3);

  const int stride =  in_h/grad_h;

  if (chunks_count == 0){
      chunks_count = int(sqrt(grad_h*grad_w*batch_size));
      if (chunks_count % batch_size > 0){
          chunks_count = chunks_count + batch_size - chunks_count % batch_size;
      }
      if (chunks_count > 1024){
          chunks_count = 1024;
      }
  }

  const int chunk_size = (grad_h*grad_w*batch_size + chunks_count - 1) / chunks_count;


  //auto grad_weights = torch::zeros_like(weights);
  auto grad_weights = torch::zeros({channel_out,channel_in, kernel_size , kernel_size }, input.device());

  const int chunks = (grad_h*grad_w*batch_size + chunk_size - 1) / chunk_size;
  assert(chunks <= 1024);


  const int threads_x = (1024 / chunks);
  const dim3 threads(threads_x , chunks);


  const int targets = channel_out *  kernel_size * kernel_size;
  const dim3 blocks((targets + threads_x - 1) / (threads_x), channel_in);



  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "conv2d_cuda_backward_kernel_filter", ([&] {
    conv2d_cuda_backward_kernel_filter<scalar_t><<<blocks, threads>>>(
        grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_weights.data<scalar_t>(),  // grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        stride, chunk_size, chunks, bit_mask,
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
    int stride, int chunk_size, int chunks, int bit_mask,
    int batch_size,
    int channel_in,
    int channel_out,
    int in_h,
    int in_w,
    int kernel_size) {


    __shared__ scalar_t reduction_vec[1024];

    const int  chunk_id = threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    const int c_in = i % channel_in;
    i = i / channel_in;
    const int h = (i % in_h);
    i = i / in_h;
    const int w = (i % in_w);
    i = i / in_w;

    if (!i){
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

        if (chunks > 1){

            reduction_vec[ threadIdx.x*chunks + (threadIdx.y)] = accumulator;

            __syncthreads();

            if (threadIdx.y == 0){
                for (int i = chunks-1; i > 0; --i){
                    accumulator += reduction_vec[ threadIdx.x*chunks + i];
                    (*paccumulator) = (*paccumulator) & bit_mask;
                }
                grad_out[out_idx] = accumulator;
            }
        } else{
            grad_out[out_idx] = accumulator;
        }
        
    }
}


torch::Tensor conv2d_cuda_input(
    torch::Tensor grad_in,
    torch::Tensor weights,
    int stride, int chunks_count, int bit_mask=-1){



  const auto grad_h = grad_in.size(2);
  const auto grad_w = grad_in.size(3);

  const auto batch_size = grad_in.size(0);
  const auto in_h = stride*grad_h;
  const auto in_w = stride*grad_w;

  const auto channel_in = weights.size(1);
  const auto channel_out = weights.size(0);
  const auto kernel_size = weights.size(2);

  if (chunks_count == 0){
      chunks_count = int(sqrt(channel_out*kernel_size*kernel_size));
      if (chunks_count > 1024){
          chunks_count = 1024;
      }
  }

  const int chunk_size = (channel_out + chunks_count - 1) / chunks_count;


  auto grad_out = torch::zeros({batch_size,channel_in, in_h , in_w }, grad_in.device());


  const int chunks = (channel_out + chunk_size - 1) / chunk_size;

  assert(chunks <= 1024);


  const int threads_x = (1024 / chunks);
  const dim3 threads(threads_x , chunks);

  const int targets = channel_in * in_h * in_w * chunks;


  const dim3 blocks((targets + threads_x - 1) / (threads_x), batch_size);



  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "conv2d_cuda_backward_kernel_input", ([&] {
    conv2d_cuda_backward_kernel_input<scalar_t><<<blocks, threads>>>(
        grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_out.data<scalar_t>(),  // grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        stride, chunk_size, chunks, bit_mask,
        batch_size,
        channel_in,
        channel_out,
        in_h,
        in_w,
        kernel_size);
  }));


  return grad_out;
}


template <typename scalar_t>
__global__ void bitmask_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tensor,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tensor_out,
    int bit_mask, int workload, int maxwork) {



    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    float value;
    int* pvalue = (int*) &value;
    int max_num = (j+1)*workload;
    if (max_num > maxwork){
        max_num = maxwork;
    }

    for (int k = j*workload; k < max_num; ++k){
        value =  tensor[i][k];
        (*pvalue) = (*pvalue) & bit_mask;
        tensor_out[i][k] = value;
    }

}

torch::Tensor bitmask_cuda(
    torch::Tensor tensor, int bit_mask=-1, int workload = 1){

  const auto x = tensor.size(0);
  const auto y = tensor.size(1);

  auto tensor_out = torch::zeros({x,y}, tensor.device());

  const int threads = 1024;



  const dim3 blocks((x + threads - 1) / (threads), y/workload);
  
  AT_DISPATCH_FLOATING_TYPES(tensor.type(), "bitmask_cuda_kernel", ([&] {
    bitmask_cuda_kernel<scalar_t><<<blocks, threads>>>(
        tensor.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        tensor_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        bit_mask,workload , y);
  }));


  return tensor_out;
}