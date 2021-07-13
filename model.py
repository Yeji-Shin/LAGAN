import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        #x = torch.cat([x, self.main(x)], 1)
        #print(x.size())
        #return x
        return x + self.main(x)
#a = torch.randn((1,3,10,10))

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Bottleneck(nn.Module):
    def __init__(self, dim_in, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(dim_in)
        self.conv1 = nn.Conv2d(dim_in, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out
'''
class tmpBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(tmpBlock, self).__init__()
        tmp_out=2*dim_out
        self.main = nn.Sequential(
            nn.InstanceNorm2d(num_features=dim_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, tmp_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(num_features=tmp_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_out,dim_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, out], 1)
#        out = self.main(x)
 #       out = torch.cat([x, out], 1)
  #      out = F.avg_pool2d(out, 2)
   #     return out
        # return torch.squeeze(F.avg_pool2d(torch.cat([x, out], 1), 2),2)
        #return x + self.main(x)


class tmpBlock2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(tmpBlock2, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        out = x+self.main(x)
        # out = F.avg_pool2d(out, 2)
        return out
'''

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=32, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(6+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))  # added
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            # layers.append(Bottleneck(dim_in=curr_dim,growthRate=40))
            # curr_dim = curr_dim + 40

        # Up-sampling layers.-
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=5, stride=1, padding=2, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)


    def forward(self, x, c, ld):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)    # reshape tensor
        c = c.repeat(1, 1, x.size(2), x.size(3))  # copy tensor and repeat - for dim=0 repeat 1 time
        ld = ld.permute(2, 0, 1)  # added for ld_guided
        ld = ld.view(1, ld.size(0), ld.size(1), ld.size(2))  # added for ld_guided
        ld = ld.repeat(c.size(0), 1, 1, 1)  # added for ld_guided
        # x = torch.cat([x, c], dim=1)  # for original stargan
        x = torch.cat([x, c, ld], dim=1)  # added for ld_guided
        return self.main(x)



class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=6, stride=2, padding=2))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.BatchNorm2d(curr_dim*2))  # added
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


# added for ld_guided_localD
class localD(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, locald_conv_dim=64, locald_repeat_num=4):
        super(localD, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, locald_conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = locald_conv_dim
        for i in range(1, locald_repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=1, padding=1))
            # layers.append(nn.BatchNorm2d(curr_dim*2))  # added
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        return out_src