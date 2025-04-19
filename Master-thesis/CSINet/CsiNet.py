class RefineNet(nn.Module):
    def __init__(self, img_channels=2):
        super(RefineNet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(img_channels, 8, kernel_size=(7, 7),padding=(3,3)),
                                  nn.BatchNorm2d(8, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv2d(8, 16, kernel_size=(7, 7),padding=(3,3)),
                                  nn.BatchNorm2d(16, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv2d(16, 2, kernel_size=(7, 7),padding=(3,3)),
                                  nn.BatchNorm2d(2, eps=1e-03, momentum=0.99))

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        ori_x = x.clone()

        # concatenate
        x = self.conv(x) + ori_x

        return self.leakyRelu(x)


class CsiNet(nn.Module):
    def __init__(self, img_height=32, img_width=32, img_channels=2, residual_num=2, encoded_dim=512):
        super(CsiNet, self).__init__()

        img_total = img_height * img_width * img_channels

        self.conv1 = nn.Sequential(nn.Conv2d(img_channels, 2, kernel_size=(7, 7),padding=(3,3)),
                                   nn.BatchNorm2d(2, eps=1e-03, momentum=0.99),
                                   nn.LeakyReLU(negative_slope=0.3))

        self.dense = nn.Sequential(nn.Linear(img_total, encoded_dim),
                                   nn.Tanh(),
                                   #nn.Sigmoid(),
                                   nn.Linear(encoded_dim, img_total))

        self.decoder = nn.ModuleList([RefineNet(img_channels)
                                      for _ in range(residual_num)])

        self.conv2 = nn.Sequential(nn.Conv2d(img_channels, 2, kernel_size=(7, 7),padding=(3,3)),
                                   nn.Sigmoid())

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Encoder, convolution & reshape
        x = self.conv1(x)
        x = x.contiguous().view(batch_size, channels * height * width)

        # Dense & reshape
        x = self.dense(x)
        x = x.contiguous().view(batch_size, channels, height, width)

        # Residual decoders
        for layer in self.decoder:
            x = layer(x)

        # Final convolution
        x = self.conv2(x)

        # x = self.Encoder(x)
        # x = self.Decoder(x)

        return x