import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Iterable as _Iterable
from collections import Callable as _Callable

from .op import *
from .utils import *
from .utils import _pair

from .thirdparty._switch_norm import SwitchNorm1d, SwitchNorm2d
from .thirdparty import _sn_layers


'''
注意写 torch.jit.script 时需要手动添加非 Tensor 参数的注释
'''


class Identity(torch.jit.ScriptModule):
    '''
    torch 居然没有 identity 层
    '''
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        return x


class Upsample(torch.jit.ScriptModule):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()

        # scale_factor 不允许是整数，有点坑。。
        if isinstance(scale_factor, _Iterable):
            scale_factor = tuple([float(i) for i in scale_factor])
        else:
            scale_factor = float(scale_factor)

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    @torch.jit.script_method
    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class UpsamplingConcat(torch.jit.ScriptModule):
    __constants__ = ['method', 'align_corners']

    def __init__(self, method='bilinear', align_corners=True):
        super().__init__()
        self.method = method
        self.align_corners = align_corners

    @torch.jit.script_method
    def forward(self, x, shortpoint):
        shape = shortpoint.shape
        x = F.interpolate(x, (shape[2], shape[3]), mode=self.method, align_corners=self.align_corners)
        x = torch.cat((x, shortpoint), 1)
        return x


class Dense(torch.jit.ScriptModule):
    def __init__(self, in_feat, out_feat, act=None, bias=True, *, norm_layer_kwargs={}):
        super().__init__()

        layers = []
        den = nn.Linear(in_feat, out_feat, bias=bias is True)
        layers.append(den)

        if isinstance(bias, _Callable):
            layers.append(bias(out_feat, **norm_layer_kwargs))

        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class Dense_SN(nn.Module):
    def __init__(self, in_feat, out_feat, act=None, bias=True, *, norm_layer_kwargs={}):
        super().__init__()

        layers = []
        den = _sn_layers.SNLinear(in_feat, out_feat, bias=bias is True)

        layers.append(den)

        if isinstance(bias, _Callable):
            layers.append(bias(out_feat, **norm_layer_kwargs))
        
        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class _base_conv_setting(torch.jit.ScriptModule):
    # 用于初始化和保存卷积设置
    def __init__(self, in_ch, out_ch, ker_sz, stride, pad, act, bias, dila):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ker_sz = ker_sz
        self.stride = stride
        self.act = act
        self.bias = bias
        self.dila = dila
        self.pad = get_padding_by_name(ker_sz, pad)


class Conv2D_SN(_base_conv_setting):
    def __init__(self, in_ch, out_ch, ker_sz=3, stride=1, pad='same', act=None, bias=True, groups=1, dila=1, *, use_fixup_init=False, norm_kwargs={}):
        super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, bias, dila)

        layers = []
        conv = _sn_layers.SNConv2d(in_ch, out_ch, ker_sz, stride, pad=self.pad, dila=dila,
                                   groups=groups, bias=bias is True)

        if use_fixup_init:
            fixup_init(conv.weight.data, _pair(ker_sz), out_ch)
            if not bias:
                conv.bias.zero_()

        layers.append(conv)

        if isinstance(bias, _Callable):
            layers.append(bias(out_ch, **norm_kwargs))

        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class Conv2D_Fixup(_base_conv_setting):
    def __init__(self, in_ch, out_ch, ker_sz=3, stride=1, pad='same', act=None, groups=1, dila=1):
        super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, False, dila)

        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.multiplicator = nn.Parameter(torch.ones(1))

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ker_sz, stride=stride, padding=self.pad,
                         dilation=dila, groups=groups, bias=False)
        if act is None:
            self.act = Identity()

    @torch.jit.script_method
    def forward(self, x):
        y = x + self.bias1
        y = self.act(self.conv1(y))
        y = self.multiplicator * (y + self.bias2)
        return y


class Conv2D(_base_conv_setting):
    def __init__(self, in_ch, out_ch, ker_sz=3, stride=1, pad='same', act=None, bias=True, groups=1, dila=1, *, use_fixup_init=False, norm_kwargs={}):
        super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, bias, dila)

        layers = []
        conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ker_sz, stride=stride, padding=self.pad,
                         dilation=dila, groups=groups, bias=bias is True)

        if use_fixup_init:
            fixup_init(conv.weight.data, _pair(ker_sz), out_ch)
            if bias is True:
                conv.bias.data.zero_()

        layers.append(conv)

        if isinstance(bias, _Callable):
            layers.append(bias(out_ch, **norm_kwargs))

        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class DeConv2D(_base_conv_setting):
    def __init__(self, in_ch, out_ch, ker_sz=3, stride=1, pad='same', act=None, bias=True, groups=1, dila=1, *, use_fixup_init=False, norm_kwargs={}):
        super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, bias, dila)

        layers = []
        conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ker_sz, stride=stride,
                                  padding=self.pad, output_padding=stride-1, dilation=dila, groups=groups,
                                  bias=bias is True)

        if use_fixup_init:
            fixup_init(conv.weight.data, _pair(ker_sz), out_ch)
            if bias is True:
                conv.bias.zero_()

        layers.append(conv)

        if isinstance(bias, _Callable):
            layers.append(bias(out_ch, **norm_kwargs))

        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class DwConv2D(_base_conv_setting):
    def __init__(self, in_ch, depth_multiplier=1, ker_sz=3, stride=1, pad='same', act=None, bias=False, dila=1, *, use_fixup_init=False, norm_kwargs={}):
        out_ch = in_ch * depth_multiplier
        super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, bias, dila)

        layers = []
        conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ker_sz, stride=stride, padding=self.pad,
                         dilation=dila, groups=in_ch, bias=bias is True)

        if use_fixup_init:
            fixup_init(conv.weight.data, _pair(ker_sz), out_ch)
            if bias is True:
                conv.bias.zero_()

        layers.append(conv)

        if isinstance(bias, _Callable):
            layers.append(bias(out_ch, **norm_kwargs))

        if act:
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class AvgPool2D(torch.jit.ScriptModule):
    def __init__(self, ker_sz=3, stride=2, pad='same', ceil_mode=False, count_include_pad=True):
        super().__init__()
        pad = get_padding_by_name(ker_sz, pad)
        self.pool = nn.AvgPool2d(kernel_size=ker_sz, stride=stride, padding=pad, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.pool(inputs)
        return outputs


class MaxPool2D(torch.jit.ScriptModule):
    def __init__(self, ker_sz=3, stride=2, pad='same', ceil_mode=False):
        super().__init__()
        pad = get_padding_by_name(ker_sz, pad)
        self.pool = nn.MaxPool2d(ker_sz=ker_sz, stride=stride, padding=pad, ceil_mode=ceil_mode)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.pool(inputs)
        return outputs


class MinibatchStdDev(torch.jit.ScriptModule):
    def __init__(self, group_size: int, num_new_features: int):
        """
        constructor for the class
        """
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    @torch.jit.script_method
    def forward(self, x):
        y = minibatch_stddev(x, group_size=self.group_size, num_new_features=self.num_new_features)
        return y


class AdaIN(torch.jit.ScriptModule):
    def __init__(self):
        """
        constructor for the class
        """
        super().__init__()

    @torch.jit.script_method
    def forward(self, x, style_x):
        y = adaptive_instance_normalization(x, style_x)
        return y
