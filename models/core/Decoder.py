import torch
import torch.nn as nn
from Channel_Variations import *
from ConvLayer import *
from Classifier import *
from Upsampling import *
class Decoder(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 1000, num_blocks = 5, block_depth = 5, mode = 'sub-pixel'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.mode = mode
        self.dropOut1 = nn.Dropout(0.2)
        self.block_channels_variation = Channel_Variations().get(in_channels = self.in_channels, n_blocks = self.num_blocks, depth = block_depth)

    def build(self, in_channels = 3, num_blocks = 5, block_depth = 5, skip = None, upsampling_mode = 'sub-pixel'):
        blocks_channel_list= self.naive_block_channels_variation(blocks = self.fibonacci(num_blocks),  in_channels = in_channels, depth = block_depth).reverse()
        decoder = nn.ModuleList()

        if(upsampling_mode == 'sub-pixel'):
            upsampler =  Sub_Pixel_Conv
        elif(upsampling_mode == 'transpose-conv'):
            upsampler = Conv_Transpose
        elif(upsampling_mode == 'resize-conv'):
            upsampler = Resize_Conv

        if skip is not None:
            #TODO
            pass
        for block in range(num_blocks):
            idx_in = block*block_depth
            idx_out = (block+1)*block_depth
            ch_in = blocks_channel_list[idx_in]
            ch_out = blocks_channel_list[idx_out]
            decoder.append(upsampler(ch_in,ch_out,scale_factor=2))
        return decoder