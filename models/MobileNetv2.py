import torch
import torch.nn as nn
import math

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion = 1 ):
        super(InvertedResidual, self).__init__()
        assert stride in [1,2]

        hidden_dim = round(in_channels * expansion)
        self.identity = stride == 1 and in_channels == out_channels

        if 1 == expansion:
            self.conv = nn.Sequential(
                #depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = stride, padding = 1, groups = hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),
                #pointwise - linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            ) 
        else:
            self.conv = nn.Sequential(
                #pointwise
                nn.Conv2d(in_channels, hidden_dim, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),
                #depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = stride, padding = 1, groups = hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True),
                #pointwise -linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, inputs):
        if self.identity:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)


class MobileNetv2(nn.Module):
    def __init__(self, num_classes = 3):
        super(MobileNetv2, self).__init__()

        self.dropout1   = nn.Dropout(0.01)
        self.dropout2   = nn.Dropout(0.1)
        self.num_classes = num_classes
        #block 1
        self.conv_3x3_bn = nn.Sequential(
                                    nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1, bias = False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU6(inplace = True)
                                    )
        residual_block      = InvertedResidual
        # block 2   t   c   n   s
        #           1   16  1   1                                
        self.bottleneck_1   = residual_block(32, 16, 1, 1)
        # block 3
        #           6   24  2   2
        self.bottleneck_2_1 = residual_block(16, 24, 2, 6)
        self.bottleneck_2_2 = residual_block(24, 24, 1, 6)
        # block 4
        #           6   32  3   2
        self.bottleneck_3_1 = residual_block(24, 32, 2, 6)
        self.bottleneck_3_2 = residual_block(32, 32, 1, 6)
        self.bottleneck_3_3 = residual_block(32, 32, 1, 6)
        #block 5
        #           6   64  4   2
        self.bottleneck_4_1 = residual_block(32, 64, 2, 6)
        self.bottleneck_4_2 = residual_block(64, 64, 1, 6)
        self.bottleneck_4_3 = residual_block(64, 64, 1, 6)
        self.bottleneck_4_4 = residual_block(64, 64, 1, 6)
        #block 6
        #           6   96  3   1
        self.bottleneck_5_1 = residual_block(64, 96, 2, 6)
        self.bottleneck_5_2 = residual_block(96, 96, 1, 6)
        self.bottleneck_5_3 = residual_block(96, 96, 1, 6)
        #block 7
        #           6   160 3   2
        self.bottleneck_6_1 = residual_block(96, 160, 2, 6)
        self.bottleneck_6_2 = residual_block(160, 160, 1, 6)
        self.bottleneck_6_3 = residual_block(160, 160, 1, 6)
        #block 8
        #           6   320 1   1
        self.bottleneck_7_1 = residual_block(160, 320, 1, 6)
        #block 9 
        #           -   1280 1  1
        self.conv_1x1_bn    = nn.Sequential(
                                            nn.Conv2d(320, 1280, kernel_size = 1, stride = 1, padding = 0, bias = False),
                                            nn.BatchNorm2d(1280),
                                            nn.ReLU6(inplace = True)
                                            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, inputs):

        outputs = self.conv_3x3_bn(inputs)
        outputs = self.bottleneck_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_2_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_2_2(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_3_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_3_2(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_3_3(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_4_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_4_2(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_4_3(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_4_4(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_5_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_5_2(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_5_3(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_6_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_6_2(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_6_3(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.bottleneck_7_1(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.conv_1x1_bn(outputs)
        outputs = self.avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.dropout2(outputs)
        outputs = self.classifier(outputs)

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
# from torchsummary import  summary
# from ptflops import get_model_complexity_info
# """Load Cuda """
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True
# """"""""""""""""""

# f = MobileNetv2(100)
# f.to(device)
# summary(f,(3,64,64))
# macs, params = get_model_complexity_info(f, (3, 64, 64), as_strings=True,
#                                         print_per_layer_stat=False, verbose=False)
# print()
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))


