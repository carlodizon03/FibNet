import torch
import torch.nn as nn
import torch.optim as optim

from logger import logger
from data.dataloaders import VOC
from torch.utils.data import DataLoader
from metrics.score import SegmentationMetric

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_set = VOC(dataset = 'train',transforms=['brightness','normalize'])
    val_set = VOC(dataset = 'val', transforms=['brightness','normalize'])

    batch_size  =  4
    num_workers = 1
    train_loader = DataLoader(
                            dataset     = train_set,
                            shuffle     = True, 
                            num_workers = num_workers,
                            batch_size  = batch_size,
                            pin_memory  = True,
                            drop_last   = True 
                        )
    val_loader = DataLoader(
                            dataset     = val_set,
                            shuffle     = True, 
                            num_workers = num_workers,
                            batch_size  = batch_size,
                            pin_memory  = True,
                            drop_last   = True 
                        )
    loaders = {'train': train_loader, 'valid': val_loader}
    