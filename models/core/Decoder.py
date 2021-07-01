import torch
import torch.nn as nn
import math
from .Channel_Variations import Channel_Variations
from .ConvLayer import ConvLayer
from .Upsampling import *

class Decoder(nn.Module):
    """
        mode -> 'sub-pixel', 'transpose-conv','resize-conv'
    """
    def __init__(self, in_channels = 64, out_channels = 1000, num_blocks = 5, block_depth = 5, mode = 'sub-pixel', is_depthwise = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.mode = mode
        self.is_depthwise = is_depthwise
        self.dropout = nn.Dropout(0.2)
        self.block_channels_variation = Channel_Variations().get(in_channels = self.in_channels, n_blocks = self.num_blocks, depth = block_depth)[::-1] #revese the list
        self.decoder = self.build()
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
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
            decoder.append(upsampler(ch_in,ch_out,scale_factor=2,is_depthwise=self.is_depthwise))
        decoder.append(ConvLayer(ch_out, self.out_channels, stride = 1, padding = 1,is_dws=self.is_depthwise))
        return decoder

    def forward(self, input, skip = None):
        
        for block in range(self.num_blocks):
            input  = self.decoder[block](input)
            input  = self.dropout(input)
             
        return self.decoder[self.num_blocks](input)