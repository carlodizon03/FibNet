
import torch
import torch.nn as nn

class Classifier(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels = 1280):
        super().__init__()
        self.add_module('pointwise',nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 1, stride = 1, padding = 0, bias = False))
        self.add_module('pw_bn',nn.BatchNorm2d(hidden_channels))
        self.add_module('pw_relu', nn.ReLU6(inplace = True))
        self.add_module('avg_pool', nn.AdaptiveAvgPool2d((1,1)))
        self.add_module('view', out_view())
        self.add_module('linear', nn.Linear(hidden_channels, out_channels))    
    def forward(self, input):
        return super().forward(input)


class out_view(nn.Sequential):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim
    def forward(self,input):
        return input.view(input.size(0),self.dim) 

