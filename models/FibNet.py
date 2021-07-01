from os import name
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.nn.modules.container import Sequential
import torchvision.transforms.functional as TF
from itertools import islice
from collections import OrderedDict
from models.core.ConvLayer import ConvLayer
from models.core.Encoder import Encoder
from models.core.Decoder import Decoder

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 8, block_depth = 5, 
                mode = "classification", use_conv_cat = True, upsampling_mode = "sub-pixel",
                pretrained_backend = False, backend_path = None, is_depthwise = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.use_conv_cat = use_conv_cat
        self.pretrained_backend = pretrained_backend
        self.backend_path = backend_path
        self.mode = mode
        self.upsampling_mode = upsampling_mode
        self.is_depthwise = is_depthwise
        self.drop = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

        self.conv1 = ConvLayer(3,32,3,2, padding =1)
        if(mode == "segmentation"):
            self.encoder = Encoder(in_channels = 32, out_channels = self.out_channels ,num_blocks = self.num_blocks,
                                    block_depth = self.block_depth, mode = self.mode, use_conv_cat = self.use_conv_cat,
                                    is_depthwise = self.is_depthwise)

            self.decoder = Decoder(in_channels = self.encoder.block_channels_variation[-1], out_channels = self.out_channels,
                                    num_blocks = self.num_blocks, block_depth = self.block_depth, mode = self.upsampling_mode,
                                    is_depthwise = self.is_depthwise)
        elif(mode == "classification"):
            self.encoder = Encoder(in_channels = 32, out_channels = self.out_channels ,num_blocks = self.num_blocks,
                                    block_depth = self.block_depth, mode = self.mode, use_conv_cat = self.use_conv_cat,
                                    is_depthwise = self.is_depthwise)

        if(self.pretrained_backend):
                assert self.backend_path is not None, "Provide path to checkpoint or weight"
                checkpoint = torch.load(self.backend_path)['state_dict']
                self.load_state_dict(OrderedDict(islice(checkpoint.items(), 0,len(checkpoint.items())-14)), strict = False)
        else:
            self._initialize_weights()


    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.drop(inputs)
        outputs = self.encoder(inputs)
        if(self.mode == "segmentation"):
            outputs = self.decoder(outputs)
        return outputs

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


# """Load Cuda """
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# from torchsummary import summary
# from ptflops import get_model_complexity_info

# model = FibNet(in_channels = 3, out_channels = 12, num_blocks = 5, block_depth = 5, mode = "segmentation",
#                  pretrained_backend = False,upsampling_mode = "resize-conv", use_conv_cat= True, is_depthwise=True)
# model.to(device)
# summary(model, (3, 480, 480))
# macs, params= get_model_complexity_info(model, (3,   480, 480), as_strings=True,
#                                            print_per_layer_stat=False, verbose=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', float(macs[:-4])))#*1e-9))
# print('{:<30}  {:<8}'.format('Number of parameters: ', float(params[:-2])))#*1e-6))
