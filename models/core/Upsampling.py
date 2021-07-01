import torch
import torch.nn as nn
from .ConvLayer import ConvLayer


class Sub_Pixel_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Feature mapping
        self.feature_maps = nn.Sequential(
            ConvLayer(self.in_channels, self.in_channels*2, kernel_size=5, stride=1, padding=2,name='sub_pix'),
            ConvLayer(self.in_channels*2, self.in_channels, kernel_size=3, stride=1, padding=1,name='sub_pix')
        )
        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * (self.scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale_factor),
        )
        in_ch = int((self.in_channels * (self.scale_factor ** 2))/(self.scale_factor**2))
        self.conv_subpix_out = ConvLayer(in_ch, self.out_channels, kernel_size=3, stride=1,padding=1, name='sub-pix-conv-out')
    def forward(self, input, skip = None,):
        out = self.feature_maps(input)
        out = self.sub_pixel(out)
        out = self.conv_subpix_out(out)
        if skip is not None:     
            #TODO pad or not pad?             
            # if skip.size(2) == 7:
            #     out = TF.pad(out,[0,1,1,0])                       
            out = torch.cat([out, skip], 1)
           
        return out

class Conv_Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2, kernel_size = 2, stride = 2, padding = 0):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.decoder = ConvLayer(self.in_channels,self.in_channels,kernel_size=3,stride=1,padding=1, name = 'decoder')
        self.conv_transpose = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size = self.kernel_size, stride =  self.stride, padding = self.padding)

    def forward(self, input, skip = None):
        out = self.decoder(input)
        if skip is not None:
            #TODO pad or not pad?             
            out = torch.cat((out,skip),1)
        out = self.conv_transpose(out)
        
        return out

class Resize_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size = 3, stride = 1, padding = 0, mode = 'bilinear'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.mode = mode
        self.up = nn.Upsample(scale_factor = self.scale_factor, mode = self.mode)
        self.pad = nn.ReflectionPad2d(1),
        self.conv = ConvLayer(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, input, skip = None):
        out = self.up(input)
        out = self.conv(out)
        if skip is not None:
           #TODO pad or not pad?             
            out = torch.cat((out,skip),1)
        return out
