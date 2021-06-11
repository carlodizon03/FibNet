import torch
import torch.nn as nn
import math

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, name = ''):
        super().__init__()
        self.add_module(name+'conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride, bias = False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace = True) )
    def forward(self, input):
        return super().forward(input)



class dws(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('pointwise',nn.Conv2d(in_channels, out_channels , kernel_size = 1))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))
        self.add_module('relu2', nn.ReLU(inplace = True))
        self.add_module('depthwise',nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, groups = out_channels, bias = False))
        self.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.add_module('relu1', nn.ReLU())
    def forward(self, input):
        return super().forward(input)

class out_view(nn.Sequential):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim
    def forward(self,input):
        return input.view(input.size(0),self.dim) 


class classifier(nn.Sequential):
    def __init__(self, in_channels, out_channels, hidden_channels = 720):
        super().__init__()
        self.add_module('pointwise',nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 1, stride = 1, padding = 0, bias = False))
        self.add_module('pw_bn',nn.BatchNorm2d(hidden_channels))
        self.add_module('pw_relu', nn.ReLU6(inplace = True))
        self.add_module('avg_pool', nn.AdaptiveAvgPool2d((1,1)))
        self.add_module('view', out_view())
        self.add_module('linear', nn.Linear(hidden_channels, out_channels))    
    def forward(self, input):
        return super().forward(input)


class fibModule(nn.Module):
    '''
    '''
    def __init__(self, in_channels = 3, out_channels = 1000, num_blocks = 5, block_depth = 5, use_conv_cat = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.dropOut1 = nn.Dropout(0.1)
        self.dropOut2 = nn.Dropout(0.2)
        self.encoder,self.transition, self.classifier = self.build(in_channels = self.in_channels, num_blocks = self.num_blocks, block_depth = self.block_depth)

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
        blocks = [i*5 for i in blocks]
#        print(blocks)
        for block in blocks:
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list.append(val)
                ratio_ = self.logistic(3.414, ratio_)
                #1.2-3.26gmac
                depth_ -= 1
                ratio_list.append(ratio_)
        
        return channel_list   

    

    def build(self, in_channels = 3, num_blocks = 5, block_depth = 5, use_conv_cat = True):
        blocks_channel_list= self.naive_block_channels_variation(blocks = self.fibonacci(num_blocks),  in_channels = in_channels, depth = block_depth)
        encoder = nn.ModuleList()
        transition = nn.ModuleList()
        cls = nn.ModuleList()
        for block in range(num_blocks):
            
            in_channels = blocks_channel_list[block*block_depth]
            out_channels = blocks_channel_list[block*block_depth+1]

            #Conv2d to match the shape for concatenation
            if(use_conv_cat):
                encoder.append(ConvLayer(in_channels, 
                                        in_channels,
                                        name = 'block_'+str(block)+'_layer_0_cat_'))
            #use Maxpooling
            else:
                encoder.append(nn.MaxPool2d((3,3),1))

            #start of block conv
            encoder.append(ConvLayer(in_channels,
                                    out_channels, 
                                    name = 'block_'+str(block)+'_layer_0_'))
            for layer in range(1,block_depth):
                idx =  block*block_depth+layer
                in_channels = blocks_channel_list[idx] + blocks_channel_list[idx-1]
                out_channels = blocks_channel_list[idx+1]

                #TODO:determine which is more effective
                # if layer >2:
                #     kernel_size = 1
                # else:
                kernel_size = 3

                #Conv2d to match the shape for concatenation
                if(use_conv_cat):
                    encoder.append(ConvLayer(in_channels = blocks_channel_list[idx],
                                            out_channels = blocks_channel_list[idx],
                                            kernel_size= kernel_size,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_cat_'))
                #use Maxpooling
                else:
                    encoder.append(nn.MaxPool2d((3,3),1))

                encoder.append(ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size=kernel_size,
                                        name = 'block_'+str(block)+'_layer_'+str(layer)+'_'))
                #transition
                if layer == block_depth-1:
                     transition.append(ConvLayer(in_channels = blocks_channel_list[idx] + out_channels,
                                             out_channels = blocks_channel_list[(block+1)*block_depth],
                                             kernel_size = 1,
                                             stride = 1,
                                             name = 'block_'+str(block)+'_layer_'+str(layer)+'_transition_'))
                #break for the last index
                if idx +1 == block_depth * num_blocks:
                    cls.append(classifier(in_channels = blocks_channel_list[(block+1)*block_depth], out_channels = self.out_channels))
                    break

        return encoder, transition, cls
        
    def forward(self, inputs):
        x = inputs 
        for block in range(self.num_blocks):

            #fdrc
            cat_out = self.encoder[block*self.block_depth*2](x)

            #fconv
            out = self.encoder[block*self.block_depth*2+1](x)
            # out = self.dropOut1(out)
            for layer in range(1,self.block_depth):
                # print(out.shape, cat_out.shape)
                #fcat
                in2 = torch.cat((out,cat_out),1)
                
                #identity
                x = out

                #fdrc
                cat_out = self.encoder[block*self.block_depth*2+(layer*2)](x)

                #fconv
                out  = self.encoder[block*self.block_depth*2+(layer*2)+1](in2)
                # out = self.dropOut1(out)

                #identity of ld-1
                if layer == self.block_depth-1:
                    out = torch.cat((out,cat_out),1)

                    if(block == self.num_blocks-1):
                        out = self.transition[block](out)
                        out = self.dropOut1(out)

                    else:
                        x = self.transition[block](out)
                        x = self.dropOut1(x)
        return self.classifier[0](out)

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 8, block_depth = 5, pretrained = False, use_conv_cat = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.drop = nn.Dropout(0.05)
        self.conv1 = ConvLayer(3,32,3,2)
        # self.conv2 = ConvLayer(8,16,3,2)
        self.encoder = fibModule(in_channels = 32, out_channels = self.out_channels ,num_blocks = self.num_blocks, block_depth = self.block_depth, use_conv_cat = True)
        self._initialize_weights()

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        # inputs = self.drop(inputs)
        # inputs = self.conv2(inputs)
        inputs = self.drop(inputs)
        outputs = self.encoder(inputs)
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

# f = FibNet(in_channels=3,out_channels=100, num_blocks=3, block_depth=3)
# f.to(device)
# summary(f,(3,64,64))
# macs, params = get_model_complexity_info(f, (3, 64, 64), as_strings=True,
#                                         print_per_layer_stat=False, verbose=False)
# print()
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
