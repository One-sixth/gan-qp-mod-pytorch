import imageio
import os
from model_utils_torch import *


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, use_bn, act):
    layers = [nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, out_pad, use_bn, act):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, out_pad, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


def DenseBnAct(in_dim, out_dim, use_bn, act):
    layers = [nn.Linear(in_dim, out_dim, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm1d(out_dim, eps=1e-8, momentum=0.9))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        self.dense1 = DenseBnAct(z_dim, 256 * 4 * 4, True, None)

        self.conv1a = ConvBnAct(256, 512, 3, 1, 1, True, self.act)
        self.conv1b = DeConvBnAct(512, 512, 3, 2, 1, 1, True, self.act)
        self.conv2a = ConvBnAct(512, 384, 3, 1, 1, True, self.act)
        self.conv2b = DeConvBnAct(384, 384, 3, 2, 1, 1, True, self.act)
        self.conv3a = ConvBnAct(384, 256, 3, 1, 1, True, self.act)
        self.conv3b = DeConvBnAct(256, 256, 3, 2, 1, 1, True, self.act)
        self.conv4a = ConvBnAct(256, 128, 3, 1, 1, True, self.act)
        self.conv4b = DeConvBnAct(128, 128, 3, 2, 1, 1, True, self.act)
        self.conv5a = ConvBnAct(128, 64, 3, 1, 1, True, self.act)
        self.conv5b = DeConvBnAct(64, 64, 3, 2, 1, 1, True, self.act)

        self.conv6 = ConvBnAct(64, 3, 1, 1, 0, False, nn.Tanh())

    def forward(self, x):
        y = x
        y = self.dense1(y)
        y = y.reshape(y.shape[0], 256, 4, 4)
        y = self.conv1a(y)
        y = self.conv1b(y)
        y = self.conv2a(y)
        y = self.conv2b(y)
        y = self.conv3a(y)
        y = self.conv3b(y)
        y = self.conv4a(y)
        y = self.conv4b(y)
        y = self.conv5a(y)
        y = self.conv5b(y)
        y = self.conv6(y)

        return y


# 采样函数
def sample(path, n=9, z_samples=None):
    gnet.eval()
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n ** 2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = np.asarray(z_samples[[i * n + j]], np.float32)

            with torch.no_grad():
                z_sample = torch.from_numpy(z_sample).cuda()
                x_sample = gnet(z_sample)
                x_sample = x_sample.cpu().numpy()
            x_sample = x_sample.transpose((0, 2, 3, 1))
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
            j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(np.uint8)
    imageio.imwrite(path, figure)


if __name__ == '__main__':

    torch.no_grad()

    sample_dir = 'test_out3'

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    img_dim = 128
    z_dim = 128

    iters_per_sample = 1
    iter_count = 100
    n_size = 9

    gnet = GenNet().cuda()

    gnet.load_state_dict(torch.load("gnet3.pt"))
    print('Load model success')


    print('gnet params')
    print_params_size(gnet.parameters())

    for i in range(iter_count):

        gnet.eval()

        Z = np.random.randn(n_size ** 2, z_dim)

        if i > 0 and i % iters_per_sample == 0:
            sample('%s/test_%s.jpg' % (sample_dir, i), n_size, Z)
