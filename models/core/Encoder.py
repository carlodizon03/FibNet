import torch
import torch.nn as nn
from .Channel_Variations import Channel_Variations
from .ConvLayer import ConvLayer
from .Classifier import Classifier
class Encoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1000, num_blocks = 5, block_depth = 5,
                 use_conv_cat = True, mode = "classification", is_depthwise = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.use_conv_cat = use_conv_cat
        self.mode = mode
        self.is_depthwise = is_depthwise
        self.dropOut1 = nn.Dropout(0.2)
        self.block_channels_variation = Channel_Variations().get(in_channels = self.in_channels, n_blocks = self.num_blocks, depth = block_depth)

        if(self.mode == "classification"):
            print("building ", self.mode)
            self.encoder,self.transition, self.classifier = self.build()
        elif(self.mode == 'segmentation'):
            print("building ", self.mode)
            self.encoder, self.transition = self.build()
    

    def build(self):
        
        encoder = nn.ModuleList()
        transition = nn.ModuleList()
        if(self.mode == "classification"):
            cls = nn.ModuleList()
        for block in range(self.num_blocks):
            
            in_channels = self.block_channels_variation[block*self.block_depth]
            out_channels = self.block_channels_variation[block*self.block_depth+1]

            #Conv2d to match the shape for concatenation
            if(self.use_conv_cat):
                encoder.append(ConvLayer(in_channels, 
                                        in_channels,
                                        padding = 1,
                                        is_dws= self.is_depthwise,
                                        name = 'block_'+str(block)+'_layer_0_cat_'))
            #use Maxpooling
            else:
                encoder.append(nn.MaxPool2d(3,stride=1,padding = 1))

            #start of block conv
            encoder.append(ConvLayer(in_channels,
                                    out_channels, 
                                    padding = 1,
                                    is_dws= self.is_depthwise,
                                    name = 'block_'+str(block)+'_layer_0_'))
            for layer in range(1,self.block_depth):
                idx =  block*self.block_depth+layer
                in_channels = self.block_channels_variation[idx] + self.block_channels_variation[idx-1]
                out_channels = self.block_channels_variation[idx+1]

                #Conv2d to match the shape for concatenation
                if(self.use_conv_cat):
                    encoder.append(ConvLayer(in_channels = self.block_channels_variation[idx],
                                            out_channels = self.block_channels_variation[idx],
                                            padding = 1,
                                            is_dws= self.is_depthwise,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_cat_'))
                #use Maxpooling
                else:
                    encoder.append(nn.MaxPool2d(3,stride=1,padding = 1))

                encoder.append(ConvLayer(in_channels = in_channels,
                                        out_channels = out_channels,
                                        padding = 1,
                                        is_dws= self.is_depthwise,
                                        name = 'block_'+str(block)+'_layer_'+str(layer)+'_'))
                #transition
                if layer == self.block_depth-1:
                     if(block == self.num_blocks-1 and self.mode == "segmentation"):
                        transition.append(ConvLayer(in_channels = self.block_channels_variation[idx] + out_channels,
                                            out_channels = self.block_channels_variation[(block+1)*self.block_depth],
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            is_dws= self.is_depthwise,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_transition_'))
                        return encoder, transition

                     transition.append(ConvLayer(in_channels = self.block_channels_variation[idx] + out_channels,
                                            out_channels = self.block_channels_variation[(block+1)*self.block_depth],
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            is_dws= self.is_depthwise,
                                            name = 'block_'+str(block)+'_layer_'+str(layer)+'_transition_'))
                                            
                #break for the last index
                if idx +1 == self.block_depth * self.num_blocks:
                    if(self.mode == "classification"):
                        cls.append(Classifier(in_channels = self.block_channels_variation[(block+1)*self.block_depth], out_channels = self.out_channels))
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
                    # skip_list.append(cat_out)
                    #last block transition
                    if(block == self.num_blocks-1):
                        # if(self.mode == 'segmentation'):
                        #     return out
                        # else:
                            out = self.transition[block](out)
                            out = self.dropOut1(out)

                    else:
                        x = self.transition[block](out)
                        x = self.dropOut1(x)
        if(self.mode == "classification"):
            return self.classifier[0](out)
        elif(self.mode == 'segmentation'):
            #TODO:skip_list
            # for t in skip_list:
                # print(t.shape)
            return out#,skip_list
