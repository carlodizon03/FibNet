import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from logger import logger
from data.dataloaders import VOC
from torch.utils.data import DataLoader
from metrics.score import SegmentationMetric
from metrics.ClassIoU import IoU
from tqdm import tqdm
from utils import Display
from models.FibNet import FibNet
from torchsummary import summary
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_set = VOC.Generator(root_dir = "D:\Dataset\VOC\VOC2012",dataset = 'train',transforms=['brightness','normalize', 'resize'])
    val_set = VOC.Generator(root_dir = "D:\Dataset\VOC\VOC2012",dataset = 'val', transforms=['brightness','normalize', 'resize'])

    batch_size  = 4
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
    labels = train_set.voc_labels(names = True)
    num_class = len(labels)
    model = FibNet(in_channels = 3, out_channels = num_class, num_blocks = 5, block_depth = 5, mode = "segmentation",  use_conv_cat= True, pretrained = True, backend_path='weights\FibNet5x5\checkpoint.pth.tar')
    model.to(device)

    class_weights = torch.tensor(train_set.class_weighing())
    criterion = nn.CrossEntropyLoss(weight=class_weights.float()).to(device)
    metric = SegmentationMetric(num_class)
    class_iou = IoU(train_set.voc_labels(True))
    optimizer = optim.Adam(model.parameters(), lr=0.04)
    log = logger.Create('logs/segmentation')
    
    best_result = 0.0

    train_loss = []
    valid_loss = []
    train_steps = 0
    valid_steps = 0
    
    for epoch in tqdm(range(50), total = 50, desc= "Epoch"):
        train_loss = []
        valid_loss = []
        for phase in["train", "valid"]:
            if phase == "train":
                model.train()
                train_pbar = tqdm(loaders[phase], total = len(loaders[phase]), desc = "Training")
                for idx, batch in enumerate(train_pbar):
                    images, encoded_masks, _ = batch
                    images, encoded_masks = images.to(device), encoded_masks.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(images)
                        loss = criterion(preds, encoded_masks)
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        train_steps += 1
                        train_pbar.set_postfix({'Loss': np.mean(train_loss)})
                        log.train_loss(np.mean(train_loss), train_steps)

            if phase == "valid":
                model.eval()
                metric.reset()
                val_pbar = tqdm(loaders[phase], total = len(loaders[phase]), desc = "Validation")
                for idx, batch in enumerate(val_pbar):
                    images, encoded_masks, mask = batch
                    images, encoded_masks = images.to(device), encoded_masks.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(images)
                        loss = criterion(preds, encoded_masks)
                        valid_loss.append(loss.item())
                        valid_steps += 1
                        val_pbar.set_postfix({'Loss': np.mean(valid_loss)})
                        log.val_loss(np.mean(valid_loss), valid_steps)
                        metric.update(preds,encoded_masks)
                        pixAcc, mIoU = metric.get()
                        log.custom_scalar("PixelAcc/validation",pixAcc, valid_steps)
                        log.custom_scalar("mIoU/validation",mIoU, valid_steps)
                        val_pbar.set_postfix({'Loss': np.mean(valid_loss),'PixAcc': pixAcc, 'mIoU':mIoU})
                iou_dict, iou_mean = class_iou(preds,encoded_masks)
                log.val_iou_per_class(iou_dict, epoch)
                log.display_val_batch(images, mask, val_set.decode_mask(preds), train_steps, unNorm = True)
                result = (pixAcc+ mIoU)/2
                if result>best_result:
                    best_result=result
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'pixAcc': pixAcc,
                                'mIoU': mIoU,
                                'classIoU': iou_dict},
                                'checkpoints/FibNet.pt')