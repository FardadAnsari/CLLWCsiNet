class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class Encoder_Compression(nn.Module):
    def __init__(self):
        super(Encoder_Compression, self).__init__()
        self.conv=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(64,8,[1,7])),
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
    
class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class CRNet(nn.Module):
    def __init__(self, reduction=4):
        super(CRNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        #logger.info(f'reduction={reduction}')
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        #self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        self.convolutional_encoder=Encoder_Compression()
        self.decoder_get_feedback_in_UE=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(8,64,[1,7])),
            ("LeakyRelu_1",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("conv1x7",ConvBN(16,32,[1,7])),
            #("LeakyRelu_2",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            #("second_conv1x7",ConvBN(32,64,[1,7])),
             #("LeakyRelu_3",nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ]))
        #self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()
        batch_size, channels, height, width = x.size()
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        #out = self.encoder_fc(out.view(n, -1))
        out = out.contiguous().view(batch_size,64 ,1,32)
        out=self.convolutional_encoder(out)
        out=self.decoder_get_feedback_in_UE(out)
        out = out.contiguous().view(batch_size,64 ,1,32)
        #out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out.view(n, c, h, w))
        out = self.sigmoid(out)

        return out


def crnet(reduction=4):
    r""" Create a proposed CRNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CRNet
    """

    model = CRNet(reduction=reduction)
    return model