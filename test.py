import imageio
import os
from model_utils_torch import *
import time


class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.act2 = nn.Tanh()

        norm_layer1d = nn.BatchNorm1d
        norm_layer2d = nn.BatchNorm2d

        self.dense1 = Dense(z_dim, 512 * 4 * 4, self.act, norm_layer1d)

        self.conv1 = DeConv2D(512, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv2 = DeConv2D(256, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv3 = DeConv2D(256, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv4 = DeConv2D(128, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv5 = DeConv2D(128, 64, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv6 = Conv2D(64, 3, 5, 1, 'same', self.act2, norm_layer2d, use_fixup_init=True)

    def forward(self, x):
        y = self.dense1(x)
        y = y.reshape(y.shape[0], 512, 4, 4)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
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

    sample_dir = 'test_out'

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    img_dim = 128
    z_dim = 128

    iters_per_sample = 1
    iter_count = 100
    n_size = 9

    gnet = GenNet().cuda()


    gnet.load_state_dict(torch.load("gnet.pt"))
    print('Load model success')

    print('gnet params')
    print_params_size(gnet.parameters())

    Z = np.random.randn(n_size ** 2, z_dim)

    ts1 = time.time()

    for i in range(iter_count):

        gnet.eval()

        Z = np.random.randn(n_size ** 2, z_dim)

        if i > 0 and i % iters_per_sample == 0:
            sample('%s/test_%s.jpg' % (sample_dir, i), n_size, Z)
