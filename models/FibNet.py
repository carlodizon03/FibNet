import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.nn.modules.container import Sequential
import torchvision.transforms.functional as TF
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0,  name = ''):
        super().__init__()
        self.add_module(name+'conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, bias = False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace = True) )
    def forward(self, input):
        return super().forward(input)



class dws(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('pointwise',nn.Conv2d(in_channels, out_channels , kernel_size = 1))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))
        self.add_module('relu2', nn.ReLU6(inplace = True))
        self.add_module('depthwise',nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, groups = out_channels, bias = False))
        self.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU6(inplace = True))
    def forward(self, input):
        return super().forward(input)

class out_view(nn.Sequential):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim
    def forward(self,input):
        return input.view(input.size(0),self.dim) 


class classifier(nn.Sequential):
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

class Sub_Pixel_Conv(nn.Module):
    def __init__(self, in_channels, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (self.scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale_factor),
        )
    def forward(self, input, skip = None,):
        out = self.feature_maps(input)
        out = self.sub_pixel(out)
        if skip is not None:     
            #TODO pad or not pad?             
            # if skip.size(2) == 7:
            #     out = TF.pad(out,[0,1,1,0])                       
            out = torch.cat([out, skip], 1)
           
        return out

class Conv_Transpose(nn.Module):
    def __init__(self, in_channels, scale_factor = 2, kernel_size = 2, stride = 2, padding = 0):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.decoder = ConvLayer(self.in_channels,self.in_channels/self.scale_factor,kernel_size=3,stride=1,padding=1, name = 'decoder')
        self.conv_transpose = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size = self.kernel_size, stride =  self.stride, padding = self.padding)

    def forward(self, input, skip = None):
        out = self.decoder(input)
        out = self.conv_transpose(out)
        if skip is not None:
            #TODO pad or not pad?             
            out = torch.cat((out,skip),1)
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
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, input, skip = None):
        out = self.up(input)
        out = self.conv(out)
        if skip is not None:
           #TODO pad or not pad?             
            out = torch.cat((out,skip),1)
        return out

class decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor = 2, option = 'sub-pixel', mode = 'bilinear', stride = 2, kernel_size = 2, padding = 0,  activation = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.scale_factor = scale_factor
        self.option = option
        self.mode = mode
        if(self.option == "sub-pixel"):
            self.upsample = Sub_Pixel_Conv(self.in_channels, self.scale_factor)
        if(self.option == "resize-conv"):
            self.upsample = Resize_Conv(self.in_channels, self.out_channels, self.scale_factor, self.kernel_size, self.stride, self.padding, self.mode)
        if(self.option == "transposed"):
            self.upsample = Conv_Transpose(self.in_channels, self.scale_factor, self.kernel_size, self.stride, self.padding)
    def forward(self, input, skip = None):
        return self.upsample(input,skip)

class fibModule(nn.Module):
    '''
    '''
    def __init__(self, in_channels = 3, out_channels = 1000, num_blocks = 5, block_depth = 5, use_conv_cat = True, mode = "classification"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.use_conv_cat = use_conv_cat
        self.mode = mode
        self.dropOut1 = nn.Dropout(0.01)
        if(self.mode == "classification"):
            print("building ", self.mode)
            self.encoder,self.transition, self.classifier = self.build(in_channels = self.in_channels, num_blocks = self.num_blocks, block_depth = self.block_depth, use_conv_cat = self.use_conv_cat, mode = self.mode)
        elif(self.mode == 'segmentation'):
            print("building ", self.mode)
            self.encoder, self.transition = self.build(in_channels = self.in_channels, num_blocks = self.num_blocks, block_depth = self.block_depth, use_conv_cat = self.use_conv_cat, mode = self.mode)
    
    def fibonacci(self,depth):
        f = []
        for i in range(1,depth+1):
            num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
            den = 2**i * (math.sqrt(5))
            f.append(int(num/den))
        return(f)

    def logistic(self, a, x):
        return a * x * (1-x)

    def second_order_logistic(self, a, x):
        ''' Function to get the second order or 2 period logistic function. 
            This Function gives a dual hump nonlinear plot.
            
            Xt+2 = F[F(Xt)] = F[Xt+1] = a * Xt+1 * (1-Xt+1)

        '''
        return self.logistic(a,self.logistic(a,x))  

    def naive_block_channels_variation(self, blocks, in_channels = 5,  depth = 5, ratio = 0.618):
        channel_list =[in_channels]
        ratio_list = [ratio]
        # blocks = [i*2 for i in blocks]
#        print(blocks)
        for block in blocks:
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list.append(val)
                ratio_ = self.logistic(2.4, ratio_)
                #1.2-3.26gmac
                depth_ -= 1
                ratio_list.append(ratio_)
        # plt.plot(channel_list)
        # plt.show()
        return channel_list   

    

    def build(self, in_channels = 3, num_blocks = 5, block_depth = 5, use_conv_cat = True, mode = "classification"):
        blocks_channel_list= self.naive_block_channels_variation(blocks = self.fibonacci(num_blocks),  in_channels = in_channels, depth = block_depth)
        encoder = nn.ModuleList()
        transition = nn.ModuleList()
        if(mode == "classification"):
            cls = nn.ModuleList()
        for block in range(num_blocks):
            
            in_channels = blocks_channel_list[block*block_depth]
            out_channels = blocks_channel_list[block*block_depth+1]

            #Conv2d to match the shape for concatenation
            if(use_conv_cat):
                encoder.append(ConvLayer(in_channels, 
                                        in_channels,
                                        padding = 1,
                                        name = 'block_'+str(block)+'_layer_0_cat_'))
            #use Maxpooling
            else:
                encoder.append(nn.MaxPool2d(3,stride=1,padding = 1))

            #start of block conv
            encoder.append(ConvLayer(in_channels,
                                    out_channels, 
                                    padding = 1,
                                    name = 'block_'+str(block)+'_layer_0_'))
            for layer in range(1,block_depth):
                idx =  block*block_depth+layer
                in_channels = blocks_channel_list[idx] + blocks_channel_list[idx-1]
                out_channels = blocks_channel_list[idx+1]

                #Conv2d to match the shape for concatenation
                if(use_conv_cat):
                    encoder.append(ConvLayer(in_channels = blocks_channel_list[idx],
                                            out_channels = blocks_channel_list[idx],
                                            padding = 1,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_cat_'))
                #use Maxpooling
                else:
                    encoder.append(nn.MaxPool2d(3,stride=1,padding = 1))

                encoder.append(ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        padding = 1,
                                        name = 'block_'+str(block)+'_layer_'+str(layer)+'_'))
                #transition
                if layer == block_depth-1:
                     if(block == num_blocks-1 and mode == "segmentation"):
                        return encoder, transition
                     transition.append(ConvLayer(in_channels = blocks_channel_list[idx] + out_channels,
                                            out_channels = blocks_channel_list[(block+1)*block_depth],
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_transition_'))
                                            
                #break for the last index
                if idx +1 == block_depth * num_blocks:
                    if(mode == "classification"):
                        cls.append(classifier(in_channels = blocks_channel_list[(block+1)*block_depth], out_channels = self.out_channels))
                        return encoder, transition, cls
                    


                    
    def forward(self, inputs):
        x = inputs 
        for block in range(self.num_blocks):
            
            #fdrc
            cat_out = self.encoder[block*self.block_depth*2](x)
            if(self.use_conv_cat): 
                cat_out = self.dropOut1(cat_out)

            #fconv
            out = self.encoder[block*self.block_depth*2+1](x)
            out = self.dropOut1(out)

            for layer in range(1,self.block_depth):
                #fcat
                in2 = torch.cat((out,cat_out),1)
                
                #identity
                x = out

                #fdrc
                cat_out = self.encoder[block*self.block_depth*2+(layer*2)](x)
                if(self.use_conv_cat): 
                    cat_out = self.dropOut1(cat_out)

                #fconv
                out  = self.encoder[block*self.block_depth*2+(layer*2)+1](in2)
                out = self.dropOut1(out)

                #identity of ld-1
                if layer == self.block_depth-1:
                    #transition concat
                    out = torch.cat((out,cat_out),1)

                    #last block transition
                    if(block == self.num_blocks-1):
                        if(self.mode == 'segmentation'):
                            return out
                        else:
                            out = self.transition[block](out)
                            out = self.dropOut1(out)

                    else:
                        x = self.transition[block](out)
                        x = self.dropOut1(x)
        if(self.mode == "classification"):
            return self.classifier[0](out)
        elif(self.mode == 'segmentation'):
            return out

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 8, block_depth = 5, mode = "classification", pretrained = False, use_conv_cat = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.use_conv_cat = use_conv_cat
        self.pretrained = pretrained
        self.mode = mode
        self.drop = nn.Dropout(0.01)
        self.conv1 = ConvLayer(3,32,3,2, padding =1)
        self.encoder = fibModule(in_channels = 32, out_channels = self.out_channels ,num_blocks = self.num_blocks, block_depth = self.block_depth, mode = self.mode, use_conv_cat = self.use_conv_cat)
        self.upsample1 = decoder(242, out_channels = 121, scale_factor=2, kernel_size=3,stride=1,padding=1, option="resize-conv")
        self.upsample2 = decoder(121, out_channels = 60, scale_factor=2, kernel_size=3,stride=1,padding=1, option="resize-conv")
        self.upsample3 = decoder(60, out_channels = 30, scale_factor=2, kernel_size=3,stride=1,padding=1, option="resize-conv")
        self.upsample4 = decoder(30, out_channels = 15, scale_factor=2, kernel_size=3,stride=1,padding=1, option="resize-conv")
        self.upsample5 = decoder(15, out_channels = 3, scale_factor=2, kernel_size=3,stride=1,padding=1, option="resize-conv")

        self._initialize_weights()

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.drop(inputs)
        outputs = self.encoder(inputs)
        outputs = self.upsample1(outputs)
        outputs = self.upsample2(outputs)
        outputs = self.upsample3(outputs)
        outputs = self.upsample4(outputs)
        outputs = self.upsample5(outputs)

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

# model = FibNet(in_channels = 3, out_channels = 3, num_blocks = 5, block_depth = 5, mode = "segmentation", pretrained = False, use_conv_cat= True)
# model.to(device)
# summary(model, (3, 384, 384))
# macs, params= get_model_complexity_info(model, (3,  384, 384), as_strings=True,
#                                            print_per_layer_stat=False, verbose=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', float(macs[:-4])))#*1e-9))
# print('{:<30}  {:<8}'.format('Number of parameters: ', float(params[:-2])))#*1e-6))
