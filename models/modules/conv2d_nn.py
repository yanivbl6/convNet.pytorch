import torch

from torch.utils.cpp_extension import load
from torch.autograd import Function

from .sparse_ops import Sparse_NHWC

from scipy import special
import math
import numpy as np

global conv2d_op 
conv2d_op = load(name='conv2d', sources=['models/modules/kernels/conv2d_cuda.cpp', 'models/modules/kernels/conv2d_cuda_kernel.cu'])


def calc_bit_mask(n):
    if n <= 0:
        return -(2**(-n))
    else:
        return -(2**(23-n))

class ConvOptions():
    def __init__(self, chunk_size = 1, bits = 0, factor = 1):
        self.chunk_size = chunk_size
        self.bits_mask = calc_bit_mask(bits)
        self.bits = bits
        self.factor = 1

    def set_factor(self, factor):
        self.factor = factor

class AllConvOptions():
    def __init__(self, chunk_sizes = [1,1,1], acc_bits = [0,0,0], M=0, N=0, fix_mode = 0):
        self.conv_opts = ConvOptions(chunk_sizes[0], acc_bits[0],1) 
        self.conv_opts_i = ConvOptions(chunk_sizes[1], acc_bits[1],1) 
        self.conv_opts_w = ConvOptions(chunk_sizes[2], acc_bits[2],1) 
        self.N = N
        self.M = M
        self.fix_mode = fix_mode

    def unpack(self):
        return (self.conv_opts, self.conv_opts_i, self.conv_opts_w, self.N, self.M, self.fix_mode)

class acculmulation_conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, stride , pars = ConvOptions(), wpars = ConvOptions(), ipars = ConvOptions()):
        ctx.save_for_backward(x, w)
        ctx.saved_params = (stride, wpars, ipars)
        return conv2d_op.forward(x, w, stride, pars.chunk_size, pars.bits_mask) * pars.factor

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        stride, wpars, ipars = ctx.saved_params
        x_grad = w_grad = None
        kernel_size = w.size(2)

        if ctx.needs_input_grad[0]:
            x_grad =  conv2d_op.input_grad(grad_output, w, stride, ipars.chunk_size, ipars.bits_mask) * ipars.factor
        if ctx.needs_input_grad[1]:
            w_grad = conv2d_op.weight_grad(grad_output, x, kernel_size, wpars.chunk_size, wpars.bits_mask)  * wpars.factor
        return x_grad, w_grad, None, None, None, None
    




def sim_VRR(macc, mp, n, N=10000):
    if n<=1:
        return 1.0

    iters = int((N*n+100000-1)/100000)
    j = 0
    
    accumulator_macc = torch.zeros([N], dtype = torch.float).cuda()

    bitmask_mp = calc_bit_mask(mp)
    bitmask_macc = calc_bit_mask(macc)
    
    per_iter = int(N/iters)
    for j in range(iters):
        s = j*per_iter
        e = min((j+1)*per_iter, N)
        cnt = e-s
        
        rand_vec = torch.randn([cnt,n], dtype = torch.float).cuda()
        rand_vec = conv2d_op.bitmask(rand_vec, bitmask_mp, 1)

        for i in range(n):
            accumulator_macc[s:e] = accumulator_macc[s:e] + rand_vec[:,i]
            accumulator_macc[s:e] = conv2d_op.bitmask(accumulator_macc[s:e].unsqueeze(1), bitmask_macc, 1).squeeze()

    Var_swamping = accumulator_macc.std()


    vrr = (Var_swamping**2/n).cpu()
    return vrr
        
def Q(x):
    return 0.5 - 0.5*special.erf(x/np.sqrt(2))

def qi(macc, i):
    return 2*Q(2**macc / np.sqrt(i))*(1 - 2*Q(2**macc / np.sqrt(i-1)))
    
def alpha_jr(macc, mp, jr):
    tmp = 2**(macc - 3*mp)/3
    J2 = [(2**j) for j in range(1,jr)]
    return tmp*np.sum([ jj * (jj- 1)*(2*jj-1)  for jj in J2])

def alpha(macc, mp):
    return alpha_jr(macc, mp, mp+1)

def qjr(macc, mp, n, jr):
    N = 2**(macc -mp + jr) ## N_{jr-1}
    return N*2*Q(2**(macc-mp + jr - 1)/ np.sqrt(n))*(1-2*Q(2**(macc-mp+jr)/(np.sqrt(n))))

def k1(macc, mp, n):
    a = alpha(macc,mp)
    return np.sum([qi(macc,i)  for i in range(2,n) if i>a])

def k2(macc, mp, n):
    return np.sum([qjr(macc,mp, n, jr)  for jr in range(2,mp+1) if n > alpha_jr(macc,mp,jr)])

def k3(macc, mp, n):
    return 1 - 2*Q(2**(macc - mp + 1)/ np.sqrt(n))

def kfunc(macc, mp, n):
    return k1(macc, mp,n) + k2(macc, mp,n) + k3(macc, mp,n)

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def VRR(macc, mp, n):
    k3_ = k3(macc, mp, n)
    k = kfunc(macc, mp, n)
    a = alpha(macc, mp)
    
    tmp1 = np.sum([(i-a)*qi(macc,i) for i in range(2,n) if i > a] )
    tmp2 = np.sum([relu(n- alpha_jr(macc, mp, jr)) * qjr(macc,mp,n , jr)  for jr in range(2,mp+1)] )
    return (tmp1 + tmp2 + n*k3_)/ (n*k)
    



def fwd_accumulation(conv_opts, kernel_size, in_channels):
    accumulation_width = kernel_size*kernel_size*in_channels

    chunks = conv_opts.chunk_size
    if chunks == 0:
        chunks = int(math.sqrt(accumulation_width))
        if (chunks > 1024):
            chunks = 1024
        
    chunk_size = (in_channels + chunks - 1) // chunks
    chunks = (in_channels + chunk_size - 1) // chunk_size
    n1 = (accumulation_width+chunks-1)//chunks
    n2 = chunks

    return n1,n2


def bwd_accumulation(conv_opts, kernel_size, out_channels):
    accumulation_width = kernel_size*kernel_size*out_channels

    chunks = conv_opts.chunk_size
    if chunks == 0:
        chunks = int(math.sqrt(accumulation_width))
        if (chunks > 1024):
            chunks = 1024
        
    chunk_size = (out_channels + chunks - 1) // chunks
    chunks = (out_channels + chunk_size - 1) // chunk_size
    n1 = (accumulation_width+chunks-1)//chunks
    n2 = chunks
    
    return n1,n2

def grad_accumulation(conv_opts, batch_size, stride, h_in ,w_in ):
    accumulation_width = batch_size*(w_in/stride)*(h_in/stride)

    chunks = conv_opts.chunk_size
    if chunks == 0:
        chunks = int(math.sqrt(accumulation_width))
        if (chunks % batch_size > 0):
            chunks = chunks + batch_size - chunks % batch_size
        
        if (chunks > 1024):
            chunks = 1024
        
    chunk_size = (accumulation_width + chunks - 1) // chunks
    chunks = (accumulation_width + chunk_size - 1) // chunk_size
    n1 = (accumulation_width+chunks-1)//chunks
    n2 = chunks
    
    return int(n1),int(n2)


def simulate_factor(conv_opts, n1,n2, mp=23, mode = 1):
    macc = conv_opts.bits
    if (mode == 1 or mode ==3):
        vrr= sim_VRR(macc, mp, n1)*sim_VRR(macc, min(macc, mp + int(np.log(n2) /np.log(2))) , n2) 
    elif (mode == 2 or mode ==4):
        vrr= VRR(macc, mp, n1)*VRR(macc, min(macc, mp + int(np.log(n2) /np.log(2))) , n2) 
    else:
        raise "Unknown Mode"

    if (mode == 1 or mode ==2):
        return 1.0/np.sqrt(vrr)
    elif (mode == 3 or mode ==4):
        return np.sqrt(vrr)

class AConv2d(torch.nn.Conv2d):    
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True , conv_options = None ,  **kwargs):

        self.N = 0 
        self.M = 0

        if conv_options is None:
            self.conv_opts = ConvPars(8,0)
            self.conv_opts_i = ConvPars(8,0)
            self.conv_opts_w = ConvPars(8,0)
        elif isinstance(conv_options, AllConvOptions):
            self.conv_opts, self.conv_opts_i, self.conv_opts_w, self.N, self.M, self.fix_mode = conv_options.unpack()
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
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.has_bias = bias

        self.need_config = False

        if self.fix_mode > 0:
            self.need_config = True


    def configure_factors(self,x):
        batch_size = x.size(0)
        h_in = x.size(2)
        w_in = x.size(3)

        fwc_n1, fwc_n2 = fwd_accumulation(self.conv_opts, self._kernel_size, self._in_channels)
        bwd_n1, bwd_n2 = bwd_accumulation(self.conv_opts_i, self._kernel_size, self._out_channels)
        grad_n1, grad_n2 = grad_accumulation(self.conv_opts_w, batch_size, self._stride, h_in ,w_in)



        self.conv_opts.set_factor(simulate_factor(self.conv_opts, fwc_n1, fwc_n2,   mode = self.fix_mode))
        self.conv_opts_i.set_factor(simulate_factor(self.conv_opts_i, bwd_n1, bwd_n2,    mode = self.fix_mode))
        self.conv_opts_w.set_factor(simulate_factor(self.conv_opts_w, grad_n1, grad_n2,    mode = self.fix_mode))

        if (np.isnan(self.conv_opts_w.factor) or np.isnan(self.conv_opts_i.factor) or np.isnan(self.conv_opts.factor)):
            import pdb; pdb.set_trace()

        print(f"Conv Layer, C_in = {self._in_channels}, c_out = {self._out_channels}, h_in = {h_in}, w_in = {w_in}, stride = {self._stride}:")
        print("Factors: (fwd: %.04f, bwd: %.04f, grad: %.04f)" % (self.conv_opts.factor, self.conv_opts_i.factor, self.conv_opts_w.factor))

        self.need_config = False

    def get_sparse_weights(self):
        if self.M > 0:
            return Sparse_NHWC.apply(self.weight, self.N, self.M)
        else:
            return self.weight

    def forward(self, x):

        if self.need_config:
            self.configure_factors(x)


        w = self.get_sparse_weights()
        x = acculmulation_conv2d.apply(x, w, self._stride, self.conv_opts, self.conv_opts_w, self.conv_opts_i)
        if self.has_bias:
            x = x + self.bias
        return x