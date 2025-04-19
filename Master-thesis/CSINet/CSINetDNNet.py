class RefineNet(torch.nn.Module):
    def __init__(self, img_channels=2):
        super(RefineNet, self).__init__()

        self.conv = torch.nn.Sequential(torch.nn.Conv2d(img_channels, 8, kernel_size=(3, 3), padding=(1,1)),
                                  torch.nn.BatchNorm2d(8, eps=1e-03, momentum=0.99),
                                  torch.nn.LeakyReLU(negative_slope=0.3),
                                  torch.nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1,1)),
                                  torch.nn.BatchNorm2d(16, eps=1e-03, momentum=0.99),
                                  torch.nn.LeakyReLU(negative_slope=0.3),
                                  torch.nn.Conv2d(16, 2, kernel_size=(3, 3), padding=(1,1)),
                                  torch.nn.BatchNorm2d(2, eps=1e-03, momentum=0.99))

        self.leakyRelu = torch.nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        ori_x = x.clone()

        # concatenate
        x = self.conv(x) + ori_x

        return self.leakyRelu(x)


class CsiNet(torch.nn.Module):
    def __init__(self, img_height=32, img_width=32, img_channels=2, residual_num=2, encoded_dim=256):
        super(CsiNet, self).__init__()

        img_total = img_height * img_width * img_channels

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(img_channels, 2, kernel_size=(3, 3), padding=(1,1)),
                                   torch.nn.BatchNorm2d(2, eps=1e-03, momentum=0.99),
                                   torch.nn.LeakyReLU(negative_slope=0.3))

        self.dense = torch.nn.Sequential(torch.nn.Linear(img_total, encoded_dim),
                                   torch.nn.Tanh())
                                   #nn.Sigmoid(),
                                   #torch.nn.Linear(encoded_dim, img_total))
        
        self.batchnorm  = torch.nn.BatchNorm1d(encoded_dim)
        self.fully_connect_layer=torch.nn.Linear(encoded_dim,512)
        self.sigmoid_sparse=torch.nn.Sigmoid()
        self.fully_connect_layer_2=torch.nn.Linear(512,1024)
        self.sigmoid_sparse=torch.nn.Sigmoid()
        self.fully_connect_layer_3=torch.nn.Linear(1024,512)
        self.sigmoid_sparse=torch.nn.Sigmoid()
        self.fully_connect_layer_4=torch.nn.Linear(512,256)
        self.sigmoid_sparse=torch.nn.Sigmoid()
        self.fully_connect_layer_5=torch.nn.Linear(encoded_dim, img_total)
        
        
                                   
                                   
        
    
        

        self.decoder = torch.nn.ModuleList([RefineNet(img_channels)
                                      for _ in range(residual_num)])

        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(img_channels, 2, kernel_size=(3, 3), padding=(1,1)),
                                   torch.nn.Sigmoid())
                                   
    def add_noise(self,x):
        
        noise = x+torch.randn_like(x)
        #noise = torch.clip(noise,0.,1.)
        return noise

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Encoder, convolution & reshape
        x = self.conv1(x)
        x = x.contiguous().view(batch_size, channels * height * width)

        # Dense & reshape
        x = self.dense(x)
        x_skip = self.add_noise(x)
        x = self.batchnorm(x_skip)
        x = self.fully_connect_layer(x)
        x = self.sigmoid_sparse(x)
        x.to_sparse_csr()
        x=self.fully_connect_layer_2(x)
        x = self.sigmoid_sparse(x)
        x.to_sparse_csr()
        x = self.fully_connect_layer_3(x)
        x = self.sigmoid_sparse(x)
        x.to_sparse_csr()
        x=self.fully_connect_layer_4(x)
        x = self.sigmoid_sparse(x)
        x = x - x_skip
        x=self.fully_connect_layer_5(x)
        
        x = x.contiguous().view(batch_size, channels, height, width)

        # Residual decoders
        for layer in self.decoder:
            x = layer(x)

        # Final convolution
        x = self.conv2(x)

        #x = self.Encoder(x)
        #x = self.Decoder(x)

        return x