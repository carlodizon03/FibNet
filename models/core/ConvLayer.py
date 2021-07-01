import torch
import torch.nn as nn

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, is_dws = False,  name = ''):
        super().__init__()
        if(is_dws):
            self.add_module(name+'dws', dws(in_channels,out_channels, stride, padding))
        else:
            self.add_module(name+'conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, bias = False))
            self.add_module('bn', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU6(inplace = True) )
    def forward(self, input):
        return super().forward(input)

class dws(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride = 1, padding = 0):
        super().__init__()
        self.add_module('pointwise',nn.Conv2d(in_channels, out_channels , kernel_size = 1))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))
        self.add_module('relu2', nn.ReLU6(inplace = True))
        self.add_module('depthwise',nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = padding, groups = out_channels, bias = False))
        self.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU6(inplace = True))
    def forward(self, input):
        return super().forward(input)