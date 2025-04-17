import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding, groups=groups)),
            ('bn', nn.BatchNorm2d(out_planes, eps=1e-03, momentum=0.99))
        ]))


class RefineNet(nn.Module):
    def __init__(self, img_channels=2):
        super(RefineNet, self).__init__()

        self.conv = nn.Sequential(OrderedDict([
            ("first_conv1x7", ConvBN(img_channels, 8, [1, 7])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("second_conv1x7", ConvBN(8, 16, [1, 7])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("third_conv1x7", ConvBN(16, 2, [1, 7])),
        ]))

        self.conv_1 = nn.Sequential(OrderedDict([
            ("first_conv1x5", ConvBN(img_channels, 8, [1, 5])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("second_conv1x5", ConvBN(8, 16, [1, 5])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("third_conv1x5", ConvBN(16, 2, [1, 5])),
        ]))

        self.conv1x1 = ConvBN(4, 2, [1, 7])
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        ori_x = x.clone()
        x_1 = self.conv(x)
        x_2 = self.conv_1(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.relu(x)
        x = self.conv1x1(x)
        return self.relu(x + ori_x)


class Encoder_Compression(nn.Module):
    def __init__(self):
        super(Encoder_Compression, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(64, 32, [1, 7])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(32, 16, [1, 7])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.conv_2 = ConvBN(64, 16, [1, 7])
        self.conv_3 = ConvBN(32, 16, [1, 7])
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        x_1 = self.conv(x)
        x_2 = self.conv_2(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.relu(x)
        x = self.conv_3(x)
        return self.relu(x)


class CLLWCsiNet(nn.Module):
    def __init__(self, reduction=4, residual_num=2):
        super(CLLWCsiNet, self).__init__()
        self.encoder_p1 = nn.Sequential(OrderedDict([
            ("conv1x7", ConvBN(2, 2, [1, 7])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv7x1", ConvBN(2, 2, [7, 1])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.encoder_p2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 2, [1, 5])),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(2, 2, [5, 1])),
        ]))

        self.encoder_p3 = nn.Sequential(OrderedDict([
            ('conv1x3', ConvBN(2, 2, [1, 3])),
            ('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1', ConvBN(2, 2, [3, 1])),
        ]))

        self.con1x1 = ConvBN(6, 2, [1, 7])
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.encoder_compression = Encoder_Compression()

        self.decoder_get_feedback_in_UE = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(16, 32, [1, 7])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(32, 64, [1, 7])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.remove_AGN = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(16, 32, [1, 7])),
            ("LeakyReLU_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(32, 64, [1, 7])),
            ("LeakyReLU_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.decoder_refine_net = nn.ModuleList([RefineNet(2) for _ in range(residual_num)])

        self._last_cov = nn.Sequential(OrderedDict([
            ("last_conv", ConvBN(2, 2, [1, 7])),
            ("activation", nn.Sigmoid())
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adding_noise(self, x):
        signal_power = torch.mean(x**2)
        SNR_dB = 40
        SNR_linear = 10**(SNR_dB / 10)
        noise_power = signal_power / SNR_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

    def forward(self, x):
        batch_size = x.size(0)
        x_1 = self.encoder_p1(x)
        x_2 = self.encoder_p2(x)
        x_3 = self.encoder_p3(x)
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.relu(self.con1x1(x))
        x = x.view(batch_size, 64, 1, 32)

        x = self.encoder_compression(x)
        x_noisy_feedback = self.adding_noise(x)
        y = self.remove_AGN(x_noisy_feedback)
        x = self.decoder_get_feedback_in_UE(x)
        x = self.relu(x - y)
        x = x.view(batch_size, 2, 32, 32)

        for refine_layer in self.decoder_refine_net:
            x = refine_layer(x)

        return self._last_cov(x)
