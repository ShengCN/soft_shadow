import torch
import torch.nn as nn
import torch.nn.functional as F

from ssn_submodule import Conv, Up, Up_Stream, get_layer_info 
from params import params

class SSN_encoder(nn.Module):
    """ General Encoder """ 
    def __init__(self, in_channels=1, out_channels=512):
        super(SSN_encoder, self).__init__()
        
        norm, act = get_layer_info(32 - in_channels)
        self.in_conv = nn.Sequential(nn.Conv2d(in_channels, 32 - in_channels, kernel_size=7, padding=3, bias=True), norm, act)
        
        self.down_256_128  = Conv(32, 64, conv_stride=2)
        self.down_128_128  = Conv(64, 64, conv_stride=1)
        self.down_128_64   = Conv(64, 128, conv_stride=2)
        self.down_64_64    = Conv(128, 128, conv_stride=1)
        self.down_64_32    = Conv(128, 256, conv_stride=2)
        self.down_32_32    = Conv(256, 256, conv_stride=1)
        self.down_32_16    = Conv(256, 512, conv_stride=2)
        self.down_16_16_1  = Conv(512, 512, conv_stride=1)
        self.down_16_16_2  = Conv(512, out_channels, conv_stride=1)

    def forward(self, x):
        """
            Input: source image
            Output: features in each layer 
        """
        x1 = self.in_conv(x)  # 29 x 256 x 256
        x1 = torch.cat((x, x1), dim=1)  # 32 x 256 x 256 
        x2 = self.down_256_128(x1)  # 64 x 128 x 128
        x3 = self.down_128_128(x2)  # 64 x 128 x 128
        x4 = self.down_128_64(x3)  # 128 x 64 x 64
        x5 = self.down_64_64(x4)  # 128 x 64 x 64
        x6 = self.down_64_32(x5)  # 256 x 32 x 32
        x7 = self.down_32_32(x6)  # 256 x 32 x 32
        x8 = self.down_32_16(x7)  # 512 x 16 x 16
        x9 = self.down_16_16_1(x8)  # 512 x 16 x 16
        x10 = self.down_16_16_2(x9)  # 512 x 16 x 16

        return [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

class SSN_decoder(nn.Module):
    """ General Decoder """
    def __init__(self, in_channels=512, out_channels=1, concat_timer=2):
        super(SSN_decoder, self).__init__()
        
        self.up_16_16_1 = Conv(in_channels, 512)
        self.up_16_16_2 = Conv(512 * concat_timer, 512)
        self.up_16_32 = Up(512 * concat_timer, 128 * concat_timer)

        self.up_32_32 = Conv(128 * concat_timer + 256 * (concat_timer-1), 256)
        self.up_32_64 = Up(256 * concat_timer, 64 * concat_timer)

        self.up_64_64 = Conv(64 * concat_timer + 128 * (concat_timer-1), 128)
        self.up_64_128 = Up(128 * concat_timer, 32 * concat_timer)

        self.up_128_128 = Conv(32 * concat_timer + 64 * (concat_timer-1), 64)
        self.up_128_256 = Up(64 * concat_timer, 16 * concat_timer)

        # self.out_conv = Conv(16 * concat_timer + 32 * (concat_timer-1), out_channels, force_relu=True)

        self.shadow_out = Conv(16 * concat_timer + 32 * (concat_timer-1), out_channels, force_relu=True)
        self.touch_out = Conv(16 * concat_timer + 32 * (concat_timer-1), out_channels, force_relu=True)
    
    def forward(self, bottle_insert, encoder_features):
        """
            bottle_insert: 16 x 16 x N
        """
        def skip_link(a,b):
            return torch.cat((a,b), dim=1)

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = encoder_features
        if bottle_insert is not None:
            y = skip_link(x10, bottle_insert)
        else:
            y = x10

        y = self.up_16_16_1(y)
        
        y = skip_link(y, x9)
        y = self.up_16_16_2(y) 

        y = skip_link(y, x8)
        y = self.up_16_32(y) 
        
        y = skip_link(y, x7)
        y = self.up_32_32(y)
        
        y = skip_link(y, x6)
        y = self.up_32_64(y)

        y = skip_link(y, x5)
        y = self.up_64_64(y)

        y = skip_link(y, x4)
        y = self.up_64_128(y)

        y = skip_link(y, x3)
        y = self.up_128_128(y)

        y = skip_link(y, x2)
        y = self.up_128_256(y)
        
        y = skip_link(y, x1)
        return self.shadow_out(y), self.touch_out(y)
    
class SSN_Touch(nn.Module):
    def __init__(self):
        """ Two head architecture
                1. shadow head
                2. touch head
        """
        super(SSN_Touch, self).__init__()

        self.encoder = SSN_encoder(2, 512)
        self.decoder = SSN_decoder(512 + 256, 1) # two head decoder

        self.ibl_conv = Conv(512, 256)
    
    def forward(self, I_s, L_t):
        """
        Inputs:
            I_s has three channels: cutout, sketches, touch 
            L_t: light target
        Outputs:
            shadow prediction, touch prediction
        """
        code = self.encoder(I_s)
        
        ibl_bottle = L_t.view(-1, 512, 1, 1).repeat(1,1,16,16)
        ibl_bottle = self.ibl_conv(ibl_bottle)
        
        return self.decoder(ibl_bottle, code)

if __name__ == '__main__':
    print("SSN touch sanity check")
    mask = torch.zeros((1,1,256,256))
    touch = torch.zeros((1,1,256,256))
    ibl = torch.zeros((1,1,16,32))

    ssn = SSN_Touch()
    pred, touch_pred = ssn(torch.cat((mask, touch), dim=1), ibl)
    print('pred: {}, touch_pred: {}'.format(pred.shape, touch_pred.shape))
