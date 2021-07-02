import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.FibNet import FibNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataloaders.CamVid import Generator as CamVid
from metrics import ClassIoU
from logger import logger
from metrics.score import *
from utils import enet_weighting
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_set = CamVid(root_dir = "D:/Dataset/CamVid",dataset = 'train',transforms=['brightness','normalize', 'resize'])
    val_set = CamVid(root_dir = "D:/Dataset/CamVid", dataset = 'val',transforms=['brightness','normalize', 'resize'])

    batch_size  =  2
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

    num_class = len(train_set.labels_of_interest())

    
    model = FibNet(in_channels = 3, out_channels = 12, num_blocks = 5, block_depth = 3, mode = "segmentation",
                 pretrained_backend = False,upsampling_mode = "resize-conv", use_conv_cat= True, is_depthwise=True).to(device)

    class_weights = torch.tensor(enet_weighting.calculate(train_loader,num_class))
   
    criterion = nn.CrossEntropyLoss(weight=class_weights.float()).to(device)
    metric    = ClassIoU.IoU(class_labels_list = train_set.labels_of_interest())
    score = SegmentationMetric(num_class)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    log = logger.Create('logs/unitTest')
    best_validation_loss = 2

    train_loss = []
    valid_loss = []
    train_steps = 0
    valid_steps = 0

    for epoch in tqdm(range(500), total = 500, desc= "Epoch"):
        train_loss = []
        valid_loss = []
        for phase in["train", "valid"]:
            if phase == "train":
                model.train()
                train_pbar = tqdm(loaders[phase], total = len(loaders[phase]), desc = "Training")
                for idx, batch in enumerate(train_pbar):
                    images, encoded_masks, masks = batch
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
                        score.update(preds,encoded_masks)
                iou_dict, iou_mean = metric(preds,encoded_masks)
                pix_acc, mIoU = score.get()
                log.custom_scalar("PixelAccuracy/training", pix_acc, epoch)
                log.custom_scalar("ScoreMIOU/training", mIoU, epoch)
                log.train_iou_per_class(iou_dict, epoch)
                log.train_mIoU(iou_mean,epoch)
                log.display_train_batch(images,masks,train_set.decode_mask(preds),epoch, unNorm = True)
            if phase == "valid":
                model.eval()
                val_pbar = tqdm(loaders[phase], total = len(loaders[phase]), desc = "Validation")
                for idx, batch in enumerate(val_pbar):
                    images, encoded_masks, masks = batch
                    images, encoded_masks = images.to(device), encoded_masks.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(images)
                        loss = criterion(preds, encoded_masks)
                        valid_loss.append(loss.item())
                        valid_steps += 1
                        val_pbar.set_postfix({'Loss': np.mean(valid_loss)})
                        log.val_loss(np.mean(valid_loss), valid_steps)
                        score.update(preds,encoded_masks)
                iou_dict, iou_mean = metric(preds,encoded_masks)
                pix_acc, mIoU = score.get()
                log.custom_scalar("PixelAccuracy/validation", pix_acc, epoch)
                log.custom_scalar("ScoreMIOU/validation", mIoU, epoch)
                log.val_iou_per_class(iou_dict, epoch)
                log.val_miou(iou_mean,epoch)
                log.display_val_batch(images, masks, val_set.decode_mask(preds), epoch, unNorm = True)
                
                if np.mean(valid_loss) < best_validation_loss:
                    best_validation_loss = np.mean(valid_loss)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': best_validation_loss},
                                'weights/FIbNet_CamVid.pt')