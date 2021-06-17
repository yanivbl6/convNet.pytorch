import torch

from torch.utils.cpp_extension import load
from torch.autograd import Function

from .sparse_ops import Sparse_NHWC


global conv2d_op 
conv2d_op = load(name='conv2d', sources=['models/modules/kernels/conv2d_cuda.cpp', 'models/modules/kernels/conv2d_cuda_kernel.cu'])


def calc_bit_mask(n):
    if n <= 0:
        return -(2**(-n))
    else:
        return -(2**(23-n))

class ConvOptions():
    def __init__(self, chunk_size = 1, bits = 0):
        self.chunk_size = chunk_size
        self.bits_mask = calc_bit_mask(bits)

class AllConvOptions():
    def __init__(self, chunk_sizes = [1,1,1], acc_bits = [0,0,0], M=0, N=0):
        self.conv_opts = ConvOptions(chunk_sizes[0], acc_bits[0]) 
        self.conv_opts_i = ConvOptions(chunk_sizes[1], acc_bits[1]) 
        self.conv_opts_w = ConvOptions(chunk_sizes[2], acc_bits[2]) 
        self.N = N
        self.M = M
    def unpack(self):
        return (self.conv_opts, self.conv_opts_i, self.conv_opts_w, self.N, self.M)

class acculmulation_conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, stride , pars = ConvOptions(), wpars = ConvOptions(), ipars = ConvOptions()):
        ctx.save_for_backward(x, w)
        ctx.saved_params = (stride, wpars, ipars)
        return conv2d_op.forward(x, w, stride, pars.chunk_size, pars.bits_mask)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        stride, wpars, ipars = ctx.saved_params
        x_grad = w_grad = None
        kernel_size = w.size(2)

        if ctx.needs_input_grad[0]:
            x_grad =  conv2d_op.input_grad(grad_output, w, stride, ipars.chunk_size, ipars.bits_mask)
        if ctx.needs_input_grad[1]:
            w_grad = conv2d_op.weight_grad(grad_output, x, kernel_size, wpars.chunk_size, wpars.bits_mask)
        return x_grad, w_grad, None, None, None, None
    

class AConv2d(torch.nn.Conv2d):    
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True , conv_options = None ,  **kwargs):

        self.N = 0 
        self.M = 0

        if conv_options is None:
            self.conv_opts = ConvPars(8,0)
            self.conv_opts_i = ConvPars(8,0)
            self.conv_opts_w = ConvPars(8,0)
        elif isinstance(conv_options, AllConvOptions):
            self.conv_opts, self.conv_opts_i, self.conv_opts_w, self.N, self.M = conv_options.unpack()
        else:
            if len(conv_options) == 3:
                self.conv_opts, self.conv_opts_i, self.conv_opts_w = conv_options
            else:
                self.conv_opts = conv_options
                self.conv_opts_i = conv_options
                self.conv_opts_w = conv_options

        padding = (kernel_size-1)//2
        super(AConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias, 'zeros', **kwargs)
        self._stride = stride
        self.has_bias = bias

    def get_sparse_weights(self):
        if self.M > 0:
            return Sparse_NHWC.apply(self.weight, self.N, self.M)
        else:
            return self.weight

    def forward(self, x):
        w = self.get_sparse_weights()
        x = acculmulation_conv2d.apply(x, w, self._stride, self.conv_opts, self.conv_opts_w, self.conv_opts_i)
        if self.has_bias:
            x = x + self.bias
        return x