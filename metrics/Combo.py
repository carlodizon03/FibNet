import torch.nn as nn
import torch.nn.functional as F
from Dice import DiceLoss

class Combo_Dice(nn.Module):

    def __init__(self):
        super(Combo_Dice, self).__init__()

        self.dice = DiceLoss()

    def forward(self, pred, target, ce_weight = 0.5):
        
        ce = F.cross_entropy(pred,target)
