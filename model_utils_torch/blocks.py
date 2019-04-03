from .layers import *
from .op import *


class _base_resblock_setting(torch.jit.ScriptModule):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.act = act


class _base_dpnblock_setting(torch.jit.ScriptModule):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, is_first):
        super().__init__()
        self.in_ch = in_ch
        self.ori_channels = ori_channels
        self.stride = stride
        self.act = act
        self.channels_increment = channels_increment
        self.is_first = is_first
        self.out_ch = in_ch + channels_increment


class resblock_1(_base_resblock_setting):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1 or in_ch != out_ch:
            self.shortcut = Conv2D(in_ch, out_ch, 3, stride, 'same', None, True)
        else:
            self.shortcut = Identity()
        self.conv1 = Conv2D(in_ch, in_ch, 3, stride, 'same', act, True)
        self.conv2 = Conv2D(in_ch, out_ch, 3, 1, 'same', act, True)

    @torch.jit.script_method
    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = shortcut + outputs
        return outputs


class resblock_2(_base_resblock_setting):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1 or in_ch != out_ch:
            self.shortcut = Conv2D(in_ch, out_ch, 3, stride, 'same', None, True)
        else:
            self.shortcut = Identity()
        self.conv1 = Conv2D(in_ch, in_ch, 1, 1, 'same', None, True)
        self.conv2 = DwConv2D(in_ch, 1, 3, stride, 'same', act, True)
        self.conv3 = Conv2D(in_ch, out_ch, 1, 1, 'same', act, True)

    @torch.jit.script_method
    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = shortcut + outputs
        return outputs


class dpnblock_8(_base_dpnblock_setting):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, groups, is_first, **kwargs):
        super().__init__(in_ch, ori_channels, stride, act, channels_increment, is_first)
        if np.max(stride) > 1 or is_first: # input_shape[1] != self.filters:
            self.shortcut = Conv2D(in_ch, ori_channels, 3, stride, 'same', None, True)
        else:
            self.shortcut = Identity()
        self.conv1 = Conv2D(in_ch, ori_channels, 1, 1, 'same', None, True)
        # self.conv2 = depthwiseconv2d(ori_channels, 1, 3, stride, 'same', act, True)
        self.conv2 = Conv2D(ori_channels, ori_channels, 3, stride, 'same', act, True, groups=groups)
        self.conv3 = Conv2D(ori_channels, ori_channels+channels_increment, 1, 1, 'same', act, True)

    @torch.jit.script_method
    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        res_outputs, dp_outputs = outputs.split([self.ori_channels, self.channels_increment], 1)
        shortcut_res, shortcut_dp = shortcut.split([self.ori_channels, self.in_ch-self.ori_channels], 1)
        shortcut_res = shortcut_res + res_outputs
        if int(shortcut_dp.shape[1]) == 0:
            shortcut_dp = dp_outputs
        else:
            shortcut_dp = torch.cat([shortcut_dp, dp_outputs], 1)
        outputs = torch.cat([shortcut_res, shortcut_dp], 1)
        return outputs


class dpnblock_8_2(_base_dpnblock_setting):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, groups, is_first, **kwargs):
        super().__init__(in_ch, ori_channels, stride, act, channels_increment, is_first)
        if np.max(stride) > 1 or is_first: # input_shape[1] != self.filters:
            self.shortcut = Conv2D(in_ch, ori_channels, 3, stride, 'same', None, True)
        else:
            self.shortcut = Identity()
        self.conv1 = Conv2D(in_ch, ori_channels, 1, 1, 'same', None, True)
        self.conv2 = DwConv2D(ori_channels, 1, 3, stride, 'same', act, True)
        # self.conv2 = Conv2D(ori_channels, ori_channels, 3, stride, 'same', act, True, groups=groups)
        self.conv3 = Conv2D(ori_channels, ori_channels+channels_increment, 1, 1, 'same', act, True)

    @torch.jit.script_method
    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        res_outputs, dp_outputs = outputs.split([self.ori_channels, self.channels_increment], 1)
        shortcut_res, shortcut_dp = shortcut.split([self.ori_channels, self.in_ch-self.ori_channels], 1)
        shortcut_res = shortcut_res + res_outputs
        if int(shortcut_dp.shape[1]) == 0:
            shortcut_dp = dp_outputs
        else:
            shortcut_dp = torch.cat([shortcut_dp, dp_outputs], 1)
        outputs = torch.cat([shortcut_res, shortcut_dp], 1)
        return outputs


class resblock_shufflenetv2(_base_resblock_setting):
    __constants__ = ['has_shortcut']

    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1:
            shortcut = DwConv2D(in_ch, 1, 3, stride, 'same', None, True)
            shortcut2 = Conv2D(in_ch, out_ch, 1, 1, 'same', act, True)
            self.shortcut = nn.Sequential(shortcut, shortcut2)
            self.has_shortcut = True
        else:
            self.shortcut = Identity()
            self.has_shortcut = False
        self.conv1 = Conv2D(in_ch, in_ch, 1, 1, 'same', act, True)
        self.conv2 = DwConv2D(in_ch, 1, 3, stride, 'same', None, True)
        self.conv3 = Conv2D(in_ch, out_ch, 1, 1, 'same', act, True)

    @torch.jit.script_method
    def forward(self, inputs):
        if self.has_shortcut:
            outputs1 = inputs
            outputs2 = self.shortcut(inputs)
        else:
            outputs1, outputs2 = inputs.chunk(2, -1)
        outputs1 = self.conv1(outputs1)
        outputs1 = self.conv2(outputs1)
        outputs1 = self.conv3(outputs1)
        outputs = outputs1 + outputs2
        outputs = channel_shuffle(outputs, 4)
        return outputs


class group_block(torch.jit.ScriptModule):
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        layers = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=out_ch, out_ch=out_ch, stride=1, act=act, **kwargs)
            layers.append(block)
        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class group_block_ed_down(torch.jit.ScriptModule):
    '''与上面不同地方在于在最后一个残差块才下采样'''
    def __init__(self, in_ch, inter_channels, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        assert blocks >= 2, 'blocks must greate or equal than 2'
        self.in_ch = in_ch
        self.out_ch = out_ch
        layers = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=inter_channels, stride=1, act=act, **kwargs)
            elif i == blocks-1:
                block = block_type(in_ch=inter_channels, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=inter_channels, out_ch=inter_channels, stride=1, act=act, **kwargs)
            layers.append(block)
        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs



class group_dpn_block(torch.jit.ScriptModule):
    def __init__(self, in_ch, stride, act, channels_increment, groups, block_type, blocks, *, out_ch=None,
                 conv_setting=None, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        if out_ch:
            self.out_ch = out_ch
        else:
            self.out_ch = in_ch + channels_increment * blocks
        layers = []
        ori_channels = in_ch
        next_channels = in_ch
        for i in range(blocks):
            block = block_type(in_ch=next_channels, ori_channels=ori_channels,
                               channels_increment=channels_increment, stride=stride if i == 0 else 1, act=act,
                               groups=groups, is_first=i==0, **kwargs)
            next_channels += channels_increment
            layers.append(block)

        if out_ch:
            conv = Conv2D(in_ch + channels_increment * blocks, out_ch, **conv_setting)
            layers.append(conv)

        self.layers = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class Hourglass4x(torch.jit.ScriptModule):
    def __init__(self, in_ch, inter_ch, out_ch, act, block_type):
        super().__init__()

        self.ds = AvgPool2D(3, 2, 'same', True, False)

        self.gp_p1_head = group_block_ed_down(in_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p2_head = group_block_ed_down(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p3_head = group_block_ed_down(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p4_head = group_block_ed_down(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)

        self.gp_p0_body = group_block_ed_down(in_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p1_body = group_block_ed_down(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p2_body = group_block_ed_down(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p3_body = group_block_ed_down(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p4_body = group_block_ed_down(inter_ch, out_ch, out_ch, 1, act, block_type, 2)

        self.gp_p1_end = group_block(out_ch, out_ch, 1, act, block_type, 1)
        self.gp_p2_end = group_block(out_ch, out_ch, 1, act, block_type, 1)
        self.gp_p3_end = group_block(out_ch, out_ch, 1, act, block_type, 1)

    @torch.jit.script_method
    def forward(self, x):
        skip0 = self.gp_p0_body(x)

        y = self.ds(x)
        y = self.gp_p1_head(y)
        skip1 = self.gp_p1_body(y)

        y = self.ds(y)
        y = self.gp_p2_head(y)
        skip2 = self.gp_p2_body(y)

        y = self.ds(y)
        y = self.gp_p3_head(y)
        skip3 = self.gp_p3_body(y)

        y = self.ds(y)
        y = self.gp_p4_head(y)
        y = self.gp_p4_body(y)
        y = resize_ref(y, skip3)
        y = y + skip3

        y = self.gp_p3_end(y)
        y = resize_ref(y, skip2)
        y = y + skip2

        y = self.gp_p2_end(y)
        y = resize_ref(y, skip1)
        y = y + skip1

        y = self.gp_p1_end(y)
        y = resize_ref(y, skip0)
        y = y + skip0

        return y


class resblock_fixup(_base_resblock_setting):
    fixup_l = 12
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__(in_ch, out_ch, stride, act)
        if act is None:
            self.act = Identity()

        if  np.max(stride) > 1:
            self.shortcut = Conv2D_Fixup(in_ch, out_ch, 3, stride)
        elif in_ch != out_ch:
            self.shortcut = Conv2D_Fixup(in_ch, out_ch, 1, stride)
        else:
            self.shortcut = Identity()

        self.multiplicator = nn.Parameter(torch.ones(1))
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.bias3 = nn.Parameter(torch.zeros(1))
        self.bias4 = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, get_padding_by_name(3, 'same'), bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, get_padding_by_name(3, 'same'), bias=False)

        fixup_init(self.conv1.weight.data, (3, 3), out_ch)
        self.conv2.weight.data.zero_()

    @torch.jit.script_method
    def forward(self, x):
        y2 = self.shortcut(x)
        y = x + self.bias1
        y = self.conv1(y) + self.bias2
        y = self.act(y) + self.bias3
        y = self.conv2(y) * self.multiplicator + self.bias4
        return y2 + y
