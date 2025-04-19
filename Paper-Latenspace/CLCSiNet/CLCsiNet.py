
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size,
                               padding=padding)),
            ('bn', nn.BatchNorm2d(out_planes,eps=1e-03, momentum=0.99))
        ]))
        
        
class RefineNet(nn.Module):
    def __init__(self, img_channels=2):
        super(RefineNet, self).__init__()
        self.conv=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(img_channels,8,[1,7])),
            ("LeakyReLU_1",nn.LeakyReLU(negative_slope=0.3)),
            ("second_Conv1x7",ConvBN(8,16,[1,7])),
            ("LeakyReLU_2",nn.LeakyReLU(negative_slope=0.3)),
            ("third_Conv1x7",ConvBN(16,2,[1,7])),    
        ]))
        
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3)
        
        
    def forward(self, x):
        ori_x = x.clone()

        # concatenate
        x = self.conv(x) + ori_x

        return self.leakyRelu(x)
    
    
    
    
class Encoder_Compression(nn.Module):
    def __init__(self):
        super(Encoder_Compression, self).__init__()
        self.conv=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(64,32,[1,7])),
            ("LeakyReLU_1",nn.LeakyReLU(negative_slope=0.3)),
            ("second_Conv1x7",ConvBN(32,16,[1,7])),
            ("LeakyReLU_2",nn.LeakyReLU(negative_slope=0.3)),
            ("third_Conv1x7",ConvBN(16,8,[1,7])),
                 
            
        ]))
        
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3)
               
    def forward(self, x):        
        # concatenate
        x = self.conv(x)
        #print("finish")
        ###### Send Feedback From User Equipement" 
        return self.leakyRelu(x)
    
        
        
        
 
    
    

class CsiNet(nn.Module):
    def __init__(self,reduction=4,residual_num=2):
        super(CsiNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        
        self.encoder = nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(2,2,[1,7])),
            ("LeakyReLU_1",nn.LeakyReLU(negative_slope=0.3)),
            ("second_conv1x7",ConvBN(2,2,[1,7])),
            ("LeakyReLU_2",nn.LeakyReLU(negative_slope=0.3)),
            ]))
        
        
        ######################## CNN base Laten Space ########################################
        self.encoder_compression=Encoder_Compression()
        
        
        
        self.decoder_get_feedback_in_UE=nn.Sequential(OrderedDict([
            ("first_conv1x7",ConvBN(8,16,[1,7])),
            ("LeakyReLU_1",nn.LeakyReLU(negative_slope=0.3)),
            ("conv1x7",ConvBN(16,32,[1,7])),
            ("LeakyReLU_2",nn.LeakyReLU(negative_slope=0.3)),
            ("second_conv1x7",ConvBN(32,64,[1,7])),
            ("LeakyReLU_3",nn.LeakyReLU(negative_slope=0.3)),
            ]))
        
        
        self.remove_AGN=nn.Sequential(OrderedDict([
            ("conv_1x7",ConvBN(8,64,[1,7])),
            ("LeakyReLu_1",nn.LeakyReLU(negative_slope=0.3))
        ]))
        
        
        self.decoder_refine_net = nn.ModuleList([RefineNet(in_channel) for _ in range(residual_num)])
        
        self._last_cov=nn.Sequential(OrderedDict([
            ("firstcov2",ConvBN(2,2,[1,7])),
            ("activation",nn.Sigmoid())
        ]))
        
        
        
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d)):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
        
    def adding_noise(self,x):
        # Compute signal power
        signal_power = torch.mean(x**2)

        # Define the desired SNR in dB (e.g., 20 dB)
        SNR_dB = 20

        # Compute the SNR in linear scale
        SNR_linear = 10**(SNR_dB / 10)

        # Compute the noise power based on the SNR
        noise_power = signal_power / SNR_linear

        # Generate Gaussian noise with the same shape as the input tensor
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        # Add the noise to the input tensor
        x_noisy = x + noise

        return x_noisy
    
    
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        #print(x.size())
        x=self.encoder(x)
        #print(x.size())
        x = x.contiguous().view(batch_size,64 ,1,32)
        #print(x.size())
        x=self.encoder_compression(x)
        #x_noisy_feedback=self.adding_noise(x)
        #y=self.remove_AGN(x_noisy_feedback)
        x=self.decoder_get_feedback_in_UE(x)
        #x=x-y
        x = x.contiguous().view(batch_size,2 ,32,32)
        for layer in self.decoder_refine_net:
            x = layer(x)
            
        x=self._last_cov(x)
       
        
        return x
            