import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules import conv, Linear
# from torch.nn.modules.utils import _pair
from .utils import *


@torch.jit.script
def channel_shuffle(x, n_group: int):
    """
    :type n_group: int
    :type x: torch.Tensor
    """
    n, c, h, w = x.shape
    x = x.reshape(-1, n_group, c // n_group, h, w)
    x = x.transpose(1, 2)
    x = x.reshape(-1, c, h, w)
    return x


@torch.jit.script
def resize_ref(x, shortpoint, method: str='bilinear', align_corners: bool=True):
    """
    :type x: torch.Tensor
    :type shortpoint: torch.Tensor
    :type method: str
    :type align_corners: bool
    """
    hw = shortpoint.shape[2:4]
    ihw = x.shape[2:4]
    if hw != ihw:
        x = torch.nn.functional.interpolate(x, hw, mode=method, align_corners=align_corners)
    return x


@torch.jit.script
def add_coord(x):
    """
    :type x: torch.Tensor
    """
    b, c, h, w = x.shape

    y_coord = torch.linspace(0, 1, h, dtype=x.dtype)
    y_coord = y_coord.reshape(1, 1, -1, 1)
    y_coord = y_coord.repeat(b, 1, 1, w)

    x_coord = torch.linspace(0, 1, w, dtype=x.dtype)
    x_coord = x_coord.reshape(1, 1, 1, -1)
    x_coord = x_coord.repeat(b, 1, h, 1)

    o = torch.cat((x, y_coord, x_coord), 1)
    return o


@torch.jit.script
def pixelwise_norm(x, eps: float=1e-8):
    """
    Pixelwise feature vector normalization.
    :param x: input activations volume
    :param eps: small number for numerical stability
    :return: y => pixel normalized activations
    """
    return x * x.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()


@torch.jit.script
def flatten(x):
    """
    """
    y = x.reshape(x.shape[0], -1)
    return y


@torch.jit.script
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])

    ss = style_feat.shape
    cs = content_feat.shape

    style_mean = style_feat.mean((2, 3), keepdim=True)
    style_std = style_feat.reshape(ss[0], ss[1], -1).std(2, unbiased=False).reshape_as(style_mean)
    content_mean = content_feat.mean((2, 3), keepdim=True)
    content_std = content_feat.reshape(cs[0], cs[1], -1).std(2, unbiased=False).reshape_as(content_mean)

    normalized_feat = (content_feat - content_mean) / (content_std + 1e-8)
    return normalized_feat * style_std + style_mean


# mod from https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
# # @torch.jit.script
# def minibatch_stddev(x, group_size: int=4, num_new_features: int=1):
#     group_size = group_size if group_size < x.shape[0] else x.shape[0]  # Minibatch must be divisible by (or smaller than) group_size.
#     s = x.shape                                             # [NCHW]  Input shape.
#     y = x.reshape(group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
#     # y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
#     y -= y.mean(dim=0, keepdim=True)                        # [GMncHW] Subtract mean over group.
#     y = y.pow(2).mean(dim=0)                                # [MncHW]  Calc variance over group.
#     y = (y + 1e-8).sqrt()                                   # [MncHW]  Calc stddev over group.
#     y = y.mean(dim=(2,3,4), keepdim=True)                   # [Mn111]  Take average over fmaps and pixels.
#     y = y.mean(dim=2)                                       # [Mn11] Split channels into c channel groups
#     # y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
#     y = y.repeat(group_size, 1, s[2], s[3])                 # [NnHW]  Replicate over group and pixels.
#     return torch.cat((x, y), dim=1)                         # [NCHW]  Append as new fmap.


@torch.jit.script
def minibatch_stddev(x, group_size: int=4, num_new_features: int=1):
    group_size = group_size if group_size < x.shape[0] else x.shape[0]
    s = x.shape
    y = x.reshape(group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = y.pow(2).mean(dim=0)
    y = (y + 1e-8).sqrt()
    y = y.mean(dim=(2,3,4), keepdim=True)
    y = y.mean(dim=2)
    y = y.repeat(group_size, 1, s[2], s[3])
    return torch.cat((x, y), dim=1)
