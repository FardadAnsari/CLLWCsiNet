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

class Encoder_Compression(nn.Module):
    def __init__(self):
        super(Encoder_Compression, self).__init__()
        self.conv=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(64,4,[1,7])),
            ("PReLU_1",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("second_Conv1x7",ConvBN(32,16,[1,7])),
            #("PReLU_2",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("third_Conv1x7",ConvBN(16,8,[1,7])),
        ]))
        
        #self.conv_2=ConvBN(64,8,[1,7])
        #self.conv_3=ConvBN(16,8,[1,7])
        
        
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
               
    def forward(self, x):        
        # concatenate
        x_1= self.conv(x)
        #x_2= self.conv_2(x)
        #x = torch.cat((x_1, x_2), dim=1)
        #x = self.LeakyRelu(x)
        #x = self.conv_3(x)
        #print("finish")
        ###### Send Feedback From User Equipement" 
        return x_1
    
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
                 reduction=4,
                 expansion=1):
        super(ACRNet, self).__init__()
        #logger.info(f"=> Model ACRNet with reduction={reduction}, expansion={expansion}")
        total_size, w, h = 2048, 32, 32

        self.encoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(in_channels, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACREncoderBlock1", ACREncoderBlock()),
            ("ACREncoderBlock2", ACREncoderBlock()),
            #("EncoderBlock2", Encoder_Compression()),
        ]))
        #self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        self.convolutional_encoder=Encoder_Compression()
        self.decoder_get_feedback_in_UE=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(4,64,[1,7])),
            ("LeakyRelu_1",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("conv1x7",ConvBN(16,32,[1,7])),
            #("LeakyRelu_2",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("second_conv1x7",ConvBN(32,64,[1,7])),
             #("LeakyRelu_3",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ]))
        
        #self.decoder_fc = nn.Linear(total_size // reduction, total_size)
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

    def forward(self, x):
        n, c, h, w = x.detach().size()
        batch_size, channels, height, width = x.size()
        out = self.encoder_feature(x)
        #out = self.encoder_fc(out.view(n, -1))
        out = out.contiguous().view(batch_size,64 ,1,32)
        out=self.convolutional_encoder(out)
        out=self.decoder_get_feedback_in_UE(out)
        #out = self.decoder_fc(out)
        out = out.contiguous().view(batch_size,64 ,1,32)
        out = self.decoder_feature(out.view(n, c, h, w))

        return out


def acrnet(expansion=1):
    r""" Create an ACRNet architecture.
    """
    model = ACRNet(expansion=expansion)
    return model