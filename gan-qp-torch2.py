# import numpy as np
import glob
import imageio
import os
# import torch
# from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from model_utils_torch import *
import cv2
import time

# torch.cuda.set_device(1)

img_dir = r'../datasets/faces/*.jpg'

# if False will use l1
use_l2 = False


class FaceDataset(Dataset):
    def __init__(self, path, img_hw=(128, 128), iter_count=1000000):
        self.imgs_path = glob.glob(path)
        self.img_hw = img_hw
        self.iter_count = iter_count

    def __getitem__(self, _):
        item = np.random.randint(0, len(self.imgs_path))
        impath = self.imgs_path[item]
        im = imageio.imread(impath)
        im = cv2.resize(im, self.img_hw, interpolation=cv2.INTER_CUBIC)
        if im.ndim == 2:
            im = np.tile(im[..., None], (1, 1, 3))
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        im = im.astype(np.float32) / 255 * 2 - 1
        im = im.transpose(2, 0, 1)
        return im

    def __len__(self):
        return self.iter_count


class DisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        norm_layer1d = nn.BatchNorm1d
        norm_layer2d = nn.BatchNorm2d

        self.conv1 = Conv2D(3, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv2 = Conv2D(128, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv3 = Conv2D(128, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv4 = Conv2D(256, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv5 = Conv2D(256+1, 512, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv6 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.dense1 = Dense(512*4*4, 1, None, False)

        # if use_low_mem_forward:
        #     self.forward = self.forward_low_mem

    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        # use minibatch_stddev
        y = minibatch_stddev(y, 4, 1)
        y = self.conv5(y)
        # y = self.conv6(y)
        y = flatten(y)
        y = self.dense1(y)

        return y


class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.act2 = TanhScale(1.1)

        norm_layer1d = nn.BatchNorm1d
        norm_layer2d = nn.BatchNorm2d

        self.dense1 = Dense(z_dim, 512*4*4, self.act, norm_layer1d)
        
        self.conv1 = Conv2D(512, 512, 5, 1, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv2 = Conv2D(512, 256, 5, 1, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv3 = Conv2D(256, 256, 5, 1, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv4 = Conv2D(256, 128, 5, 1, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv5 = Conv2D(128, 128, 5, 1, 'same', self.act, norm_layer2d, use_fixup_init=True)
        self.conv6 = Conv2D(128, 3, 1, 1, 'same', self.act2, norm_layer2d, use_fixup_init=True)
        self.upsample = Upsample(scale_factor=2., mode='bilinear', align_corners=False)

        # self.conv1 = DeConv2D(512, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv2 = DeConv2D(256, 256, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv3 = DeConv2D(256, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv4 = DeConv2D(128, 128, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv5 = DeConv2D(128, 64, 5, 2, 'same', self.act, norm_layer2d, use_fixup_init=True)
        # self.conv6 = Conv2D(64, 3, 5, 1, 'same', self.act2, norm_layer2d, use_fixup_init=True)


    def forward(self, x):

        y = self.dense1(x)
        y = y.reshape(y.shape[0], 512, 4, 4)
        # # y = self.upsample(y)
        # y = self.conv1(y)
        # # y = self.upsample(y)
        # y = self.conv2(y)
        # # y = self.upsample(y)
        # y = self.conv3(y)
        # # y = self.upsample(y)
        # y = self.conv4(y)
        # # y = self.upsample(y)
        # y = self.conv5(y)
        # y = self.conv6(y)

        y = self.upsample(y)
        y = self.conv1(y)
        y = self.upsample(y)
        y = self.conv2(y)
        y = self.upsample(y)
        y = self.conv3(y)
        y = self.upsample(y)
        y = self.conv4(y)
        y = self.upsample(y)
        y = self.conv5(y)
        y = self.conv6(y)

        return y


# 采样函数
def sample(path, n=9, z_samples=None):
    gnet.eval()
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
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


def next_data(dataloader):
    for data in dataloader:
        yield data
    return None


if __name__ == '__main__':

    sample_dir = 'samples2'
    iter_file = 'cur_it2.txt'

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    img_dim = 128
    z_dim = 128
    batch_size = 64
    n_epoch = 1000

    iters_per_sample = 100
    iter_count = 1000000
    n_size = 9

    gnet = GenNet().cuda()
    dnet = DisNet().cuda()

    gnet_optimizer = torch.optim.Adam(gnet.parameters(), lr=0.0003, betas=(0., 0.999))
    dnet_optimizer = torch.optim.Adam(dnet.parameters(), lr=0.0003, betas=(0., 0.999))

    start_it = 0

    try:
        gnet.load_state_dict(torch.load("gnet2.pt"))
        dnet.load_state_dict(torch.load("dnet2.pt"))
        gnet_optimizer.load_state_dict(torch.load("gnet_optim2.pt"))
        dnet_optimizer.load_state_dict(torch.load("dnet_optim2.pt"))
        print('Load model success')

        if os.path.isfile(iter_file):
            try:
                start_it = int(open(iter_file, 'r').read(10))
            except:
                start_it = 0

    except (FileNotFoundError, RuntimeError):
        print('Not found save model')

    print('gnet params')
    print_params_size(gnet.parameters())
    print('dnet params')
    print_params_size(dnet.parameters())

    dataset = FaceDataset(img_dir, (img_dim, img_dim), iter_count*3)
    datasetloader = DataLoader(dataset, batch_size, True, num_workers=1, timeout=10)

    Z = np.random.randn(n_size**2, z_dim)

    ts1 = time.time()

    nd = next_data(datasetloader)


    for i in range(start_it, iter_count):

        gnet.train()
        dnet.train()
        d_loss, g_loss = None, None

        for _ in range(2):
            batch_imgs = next(nd)
            batch_z = np.asarray(np.random.randn(len(batch_imgs), z_dim), np.float32)

            if batch_imgs is None:
                exit()

            real_imgs = batch_imgs.cuda()
            batch_z = torch.from_numpy(batch_z).cuda()

            dnet_optimizer.zero_grad()

            # can more fast
            with torch.no_grad():
                fake_imgs = gnet(batch_z)

            d_real_score = dnet(real_imgs)
            d_fake_score = dnet(fake_imgs)
            d_loss = d_real_score - d_fake_score
            d_loss = d_loss[:, 0]
            if use_l2:
                d_norm = 10 * torch.sqrt(torch.mean(torch.pow(real_imgs - fake_imgs, 2), dim=[1, 2, 3], keepdim=False))
            else:
                # use l1
                d_norm = 10 * torch.mean(torch.abs(real_imgs - fake_imgs), dim=[1, 2, 3], keepdim=False)

            d_loss = torch.mean(-d_loss + 0.5 * d_loss ** 2 / d_norm)

            d_loss.backward()
            dnet_optimizer.step()

        for _ in range(1):
            batch_imgs = next(nd)
            batch_z = np.asarray(np.random.randn(len(batch_imgs), z_dim), np.float32)

            if batch_imgs is None:
                exit()

            real_imgs = batch_imgs.cuda()
            batch_z = torch.from_numpy(batch_z).cuda()

            gnet_optimizer.zero_grad()
            dnet_optimizer.zero_grad()

            fake_imgs = gnet(batch_z)

            d_real_score = dnet(real_imgs)
            d_fake_score = dnet(fake_imgs)

            g_loss = torch.mean(d_real_score - d_fake_score)

            g_loss.backward()
            gnet_optimizer.step()

        d_loss, g_loss = d_loss.item(), g_loss.item()
        if np.isnan(d_loss) or np.isnan(g_loss):
            print('Found loss Nan!', 'd_loss %f' % d_loss, 'g_loss %f' % g_loss)
            raise AttributeError('Found Nan')

        if i > 0 and i % 10 == 0:
            ts2 = time.time()
            print('iter: %d, d_loss: %.4f, g_loss: %.4f' % (i, d_loss, g_loss),
                  'time: %.4f' % (ts2 - ts1))
            ts1 = ts2

        if i > 0 and i % iters_per_sample == 0:
            sample('%s/test_%s.jpg' % (sample_dir, i), n_size, Z)
            torch.save(gnet.state_dict(), 'gnet2.pt')
            torch.save(dnet.state_dict(), 'dnet2.pt')
            torch.save(gnet_optimizer.state_dict(), 'gnet_optim2.pt')
            torch.save(dnet_optimizer.state_dict(), 'dnet_optim2.pt')
            open(iter_file, 'w').write(str(i))
