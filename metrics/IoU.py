import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
class IoU(nn.Module):
    '''Calculates the IoU Score for all the predicted class 
    
    Input Parameters:
    
    inputs  - predicted tensor of size [N,C,H,W]
    targets - target masks of size [N,H,W]
    
    Output:

    IoU     - Intersection over Union of all the predicted class vs target class.
            - Calculated with smoothing factor. 
     
    '''

    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        targets = self._separate_class_to_channels(inputs,targets, 21)
        # #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # #intersection is equivalent to True Positive count
        # #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        return 1 - ((intersection + smooth)/(union + smooth))
    
    def _separate_class_to_channels(self,preds, masks, num_classes):
        
        _mask = torch.zeros(preds.shape, dtype = torch.int64).to("cuda:0")  
        _non_class = torch.tensor(0.,dtype=torch.int64).to("cuda:0")
        for mask_idx, mask in enumerate(masks):  
            for _, label in enumerate(torch.unique(mask)):
                if(label!=0):
                    _mask[mask_idx,label,:,:]  = torch.where(mask == label, mask, _non_class).squeeze(0)
        return _mask



    