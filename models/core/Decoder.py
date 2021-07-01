import torch
import torch.nn as nn
from .Channel_Variations import Channel_Variations
from .ConvLayer import ConvLayer
from .Upsampling import *

class Decoder(nn.Module):
    """
        mode -> 'sub-pixel', 'transpose-conv','resize-conv'
    """
    def __init__(self, in_channels = 64, out_channels = 1000, num_blocks = 5, block_depth = 5, mode = 'sub-pixel'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.mode = mode
        self.dropOut1 = nn.Dropout(0.2)
        self.block_channels_variation = Channel_Variations().get(in_channels = self.in_channels, n_blocks = self.num_blocks, depth = block_depth)[::-1] #revese the list
        self.decoder = self.build()

    def build(self):
        decoder = nn.ModuleList()

        if(self.mode == 'sub-pixel'):
            upsampler =  Sub_Pixel_Conv
        elif(self.mode == 'transpose-conv'):
            upsampler = Conv_Transpose
        elif(self.mode == 'resize-conv'):
            upsampler = Resize_Conv

        for block in range(self.num_blocks):
            idx_in = block*self.block_depth
            idx_out = (block+1)*self.block_depth
            ch_in = self.block_channels_variation[idx_in]
            ch_out = self.block_channels_variation[idx_out]
            decoder.append(upsampler(ch_in,ch_out,scale_factor=2))
        decoder.append(ConvLayer(ch_out, self.out_channels,padding=1))
        return decoder

    def forward(self, input, skip = None):
        
        for block in range(self.num_blocks):
            input  = self.decoder[block](input)
        return self.decoder[self.num_blocks](input)