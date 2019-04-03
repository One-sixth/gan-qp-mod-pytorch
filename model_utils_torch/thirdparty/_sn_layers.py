import torch
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair


@torch.jit.script
def _l2normalize(v, eps: float=1e-12):
    return v / (torch.norm(v) + eps)


# @torch.jit.script
def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


class SNConv2d(conv._ConvNd):

    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:
    .. math::
        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    | :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    | :attr:`pad` controls the amount of implicit zero-paddings on both
    |  sides for :attr:`pad` number of points for each dimension.
    | :attr:`dila` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dila` does.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_ch` and `out_ch` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_ch`, each input channel is convolved with its
                 own set of filters (of size `out_ch // in_ch`).
    The parameters :attr:`ker_sz`, :attr:`stride`, :attr:`pad`, :attr:`dila` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    .. note::
         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper pad.
    .. note::
         The configuration when `groups == in_ch` and `out_ch = K * in_ch`
         where `K` is a positive integer is termed in literature as depthwise convolution.
         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(in\_channels=C_{in}, out\_channels=C_{in} * K, ..., groups=C_{in})`
    Args:
        in_ch (int): Number of channels in the input image
        out_ch (int): Number of channels produced by the convolution
        ker_sz (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        pad (int or tuple, optional): Zero-pad added to both sides of the input. Default: 0
        dila (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * pad[0] - dila[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * pad[1] - dila[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_ch, in_ch, ker_sz[0], ker_sz[1])
        bias (Tensor):   the learnable bias of the module of shape (out_ch)
        W(Tensor): Spectrally normalized weight
        u (Tensor): the right largest singular value of W.
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_ch, out_ch, ker_sz, stride=1, pad=0, dila=1, groups=1, bias=True):
        ker_sz = _pair(ker_sz)
        stride = _pair(stride)
        pad = _pair(pad)
        dila = _pair(dila)
        super(SNConv2d, self).__init__(
            in_ch, out_ch, ker_sz, stride, pad, dila,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_ch).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.pad, self.dila, self.groups)


class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`
           W(Tensor): Spectrally normalized weight
           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)