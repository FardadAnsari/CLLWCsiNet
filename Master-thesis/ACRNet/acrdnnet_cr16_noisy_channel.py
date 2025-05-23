from zmq import OUT_BATCH_SIZE
from torch import nn
from collections import OrderedDict

#from utils import logger


__all__ = ["acrnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_planes,
                               out_channels=out_planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class ACRDecoderBlock(nn.Module):
    r""" Inverted residual with extensible width and group conv
    """
    def __init__(self, expansion):
        super(ACRDecoderBlock, self).__init__()
        width = 8 * expansion
        self.conv1_bn = ConvBN(2, width, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv2_bn = ConvBN(width, width, 7, groups=4 * expansion)
        self.prelu2 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv3_bn = ConvBN(width, 2, [9, 1])
        self.prelu3 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv1_bn(x))
        residual = self.prelu2(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.prelu3(identity + residual)


class ACREncoderBlock(nn.Module):
    def __init__(self):
        super(ACREncoderBlock, self).__init__()
        self.conv_bn1 = ConvBN(2, 2, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv_bn2 = ConvBN(2, 2, [9, 1])
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv_bn1(x))
        residual = self.conv_bn2(residual)

        return self.prelu2(identity + residual)


class ACRNet(nn.Module):
    def __init__(self,
                 in_channels=2,
                 reduction=4
                 , expansion=1):
        super(ACRNet, self).__init__()
        info(f"=> Model ACRNet with reduction={reduction}")
        total_size, w, h = 2048, 32, 32

        self.encoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(in_channels, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACREncoderBlock1", ACREncoderBlock()),
            ("ACREncoderBlock2", ACREncoderBlock()),
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        #self.add_noise()
        self.batchnorm  = nn.BatchNorm1d(num_features=total_size // reduction)
        self.fully_connect_layer=nn.Linear(total_size // reduction,256)
        self.sigmoid_sparse=nn.Sigmoid()
        self.fully_connect_layer_2=nn.Linear(256,512)
        self.sigmoid_sparse=nn.Sigmoid()
        self.fully_connect_layer_3=nn.Linear(512,1024)
        self.sigmoid_sparse=nn.Sigmoid()
        self.fully_connect_layer_4=nn.Linear(1024,512)
        self.sigmoid_sparse=nn.Sigmoid()
        self.fully_connect_layer_5=nn.Linear(512,256)
        self.sigmoid_sparse=nn.Sigmoid()
        self.fully_connect_layer_6=nn.Linear(256,128)




        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, in_channels, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACRDecoderBlock1", ACRDecoderBlock(expansion=expansion)),
            ("ACRDecoderBlock2", ACRDecoderBlock(expansion=expansion)),
            ("sigmoid", nn.Sigmoid())
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def add_noise(self,x):
     noise = x+torch.randn_like(x)
     #noise = torch.clip(noise,0.,1.)
     return noise

    def forward(self, x):
        n, c, h, w = x.detach().size()

        out = self.encoder_feature(x)
        out = self.encoder_fc(out.view(n, -1))
        #out=out +torch.randn(out.size())
        #adding nosie to code word
        out_skip=self.add_noise(out)
        #batch normalization
        out=self.batchnorm(out_skip)
        #fully connect layer
        out=self.fully_connect_layer(out)
        # activation function
        out=self.sigmoid_sparse(out)
        out.to_sparse_csr()
        #fully connected layer
        out=self.fully_connect_layer_2(out)
        #activation function
        out=self.sigmoid_sparse(out)
        out.to_sparse_csr()
        # fully connected layer
        out=self.fully_connect_layer_3(out)
        #activation
        out=self.sigmoid_sparse(out)
        out.to_sparse_csr()
        #fully_connected_layer
        out=self.fully_connect_layer_4(out)
        #activation
        out=self.sigmoid_sparse(out)
        out.to_sparse_csr()
        #fully_connected_layer
        out=self.fully_connect_layer_5(out)
        #activation
        out=self.sigmoid_sparse(out)
        out.to_sparse_csr()
        #fully_connected_layer
        out=self.fully_connect_layer_6(out)




        out=out-out_skip
        out = self.decoder_fc(out)
        out = self.decoder_feature(out.view(n, c, h, w))

        return out


def acrnet(reduction=4,expansion=1):
    r""" Create an ACRNet architecture.
    """
    model = ACRNet(reduction=reduction,expansion=expansion)
    return model