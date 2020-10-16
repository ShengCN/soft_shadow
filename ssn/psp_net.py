
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_x, scale_y):
        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, x):
        h, w = self.scale_y * x.size(2), self.scale_x * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class PSP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # pooling
        self.pool2 = nn.AdaptiveAvgPool2d((2,2))
        self.pool4 = nn.AdaptiveAvgPool2d((4,4))
        self.pool8 = nn.AdaptiveAvgPool2d((8,8))

        # conv -> compress channels
        avg_channel = in_channels//4
        self.conv2 = Conv(in_channels, avg_channel)
        self.conv4 = Conv(in_channels, avg_channel)
        self.conv8 = Conv(in_channels, avg_channel)
        self.conv16 = Conv(in_channels, avg_channel)

        # up sapmle -> match dimension
        self.up2 = PSPUpsample(avg_channel, avg_channel, 16//2, 16//2)
        self.up4 = PSPUpsample(avg_channel, avg_channel, 16//4, 16//4)
        self.up8 = PSPUpsample(avg_channel, avg_channel, 16//8, 16//8)

    def forward(self, x):
        x2 = self.up2(self.conv2(self.pool2(x))) 
        x4 = self.up4(self.conv4(self.pool4(x))) 
        x8 = self.up8(self.conv8(self.pool8(x))) 
        x16 = self.conv16(x)
        return  torch.cat((x2, x4, x8, x16), dim=1)