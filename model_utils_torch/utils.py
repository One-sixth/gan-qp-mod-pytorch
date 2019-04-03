import numpy as np
import torch
from collections import Iterable as _Iterable


def get_padding_by_name(ker_sz, name='same'):
    if name.lower() == 'same':
        pad = np.int32(np.array(ker_sz) // 2).tolist()
    elif name.lower() == 'valid':
        pad = 0
    else:
        raise AssertionError(': "{}" is not expected'.format(name))
    return pad


def fixup_init(w, ker_sz, out_ch, fixup_l=12):
    k = np.cumprod(ker_sz)[-1] * out_ch
    w.normal_(0, fixup_l ** (-0.5) * np.sqrt(2. / k))


def print_params_size(parameter, dtype_size=4):
    params_count = 0
    for p in parameter:
        params_count += np.prod(list(p.shape))
    print('params size %f MB' % (params_count * dtype_size / 1024 / 1024))
    return params_count


def _pair(ker_sz):
    if isinstance(ker_sz, int):
        return ker_sz, ker_sz
    elif isinstance(ker_sz, _Iterable) and len(ker_sz) == 2:
        return tuple(ker_sz)
    else:
        raise AssertionError('Wrong kernel_size')
