import torch
import torch.nn as nn
import numpy as np

class IoU(nn.Module):
    '''
      IoU, also known as Jaccard Similarity Coefficient
      
      Calculates the IoU for each class where 1 is perfect overlap 
      and zero signifying no overlap.
      
      Init:
        
        class_labels_list -> list of classess.
        with_softmax      -> set to False if your model implements 
                             SoftMax in output.

      Forward Inputs:

        preds   -> B x N x W x H
        targets -> B x N x W
      
      Forward Outputs:

        (Dict{'class':iou},mean_iou])

    '''
    def __init__(self, class_labels_list = None,  with_softmax = True):
      super(IoU, self).__init__()
      # assert class_labels_list!=None

      self.class_labels_list = class_labels_list
      self.n_classess = len(self.class_labels_list)
      self.with_softmax = with_softmax
      if self.with_softmax:
        ''''
            Use softmax if the model does not include this in the output.
            This will give the probabilities for each class summing to 1. 
        '''
        self.softmax = nn.Softmax(dim=1)


    def forward(self, preds, targets):

      if self.with_softmax:
        preds = self.softmax(preds)

      'Get the indices of maximum probabilities'  
      preds = torch.argmax(preds, dim = 1)
      
      iou_list = []
      iou_dict = {}
      for n_class in range(1,self.n_classess):
        '''iterate for all class'''
        
        y_pred = (preds == n_class)
        y_true = (targets == n_class)

        if y_true.long().sum().item() == 0:
          'if there are no matching class labes'
          class_iou = float(0.0)
        else:
          intersection = (y_pred[y_true]).long().sum().item()
          union = y_pred.long().sum().item() + y_true.long().sum().item() - intersection 
          class_iou = float(intersection)/float(union)

        iou_list.append(class_iou)
        iou_dict[self.class_labels_list[n_class]] = class_iou

      return iou_dict, np.mean(iou_list)
