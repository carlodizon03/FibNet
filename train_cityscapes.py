from os import name
from types import new_class
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from metrics import ClassIoU
from logger import logger
from metrics.score import *
from utils import enet_weighting
from models.FibNet import FibNet

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    image_transform = transforms.Compose([
                    transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    target_transform = transforms.Compose([
                    transforms.Resize((512,512))
            ])
    train_set = Cityscapes(root="D:/Dataset/cityscapes", split='train',mode='fine',target_type='semantic',
                            transform = image_transform, target_transform = target_transform)
    val_set  = Cityscapes(root="D:/Dataset/cityscapes", split='val',mode='fine',target_type='semantic',
                            transform = image_transform ,target_transform = target_transform)
    batch_size = 1
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

    labels, colors = Cityscapes.get_labels()
    n_classes = len(labels)+1
    model = FibNet(in_channels = 3, out_channels = n_classes, num_blocks = 5, block_depth = 3, mode = "segmentation",
                 pretrained_backend = False,upsampling_mode = "sub-pixel", use_conv_cat= True, is_depthwise=False).to(device)

    class_weights = torch.tensor(np.load('dsclass_weights.npy'))
    criterion = nn.CrossEntropyLoss(weight=class_weights.float()).to(device)
    metric    = ClassIoU.IoU(class_labels_list=labels)
    score = SegmentationMetric(n_classes)
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
                    images, masks = batch
                    # ax = plt.subplot(121)
                    # ay = plt.subplot(122)
                    # ax.imshow(images[0].detach().numpy().transpose((1,2,0)))
                    # ay.imshow(masks.detach().numpy().transpose((1,2,0)))
                    # plt.show()
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(images)
                        loss = criterion(preds, masks)
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        train_steps += 1
                        train_pbar.set_postfix({'Loss': np.mean(train_loss)})
                        log.train_loss(np.mean(train_loss), train_steps)
                        score.update(preds,masks)
                        if(train_steps%20 == 0):
                            iou_dict, iou_mean = metric(preds,masks)
                            pix_acc, mIoU = score.get()
                            log.custom_scalar("PixelAccuracy/training", pix_acc, train_steps)
                            log.custom_scalar("ScoreMIOU/training", mIoU, train_steps)
                            log.train_iou_per_class(iou_dict, train_steps)
                            log.train_mIoU(iou_mean,train_steps)
                            log.display_train_batch(images,masks,Cityscapes.decode_labels(preds,colors),train_steps, unNorm = True)
            
            if phase == "valid":
                model.eval()
                val_pbar = tqdm(loaders[phase], total = len(loaders[phase]), desc = "Validation")
                for idx, batch in enumerate(val_pbar):
                    images, masks = batch
                    images, masks = images.to(device), masks.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(images)
                        loss = criterion(preds, masks)
                        valid_loss.append(loss.item())
                        valid_steps += 1
                        val_pbar.set_postfix({'Loss': np.mean(valid_loss)})
                        log.val_loss(np.mean(valid_loss), valid_steps)
                        score.update(preds,masks)
                iou_dict, iou_mean = metric(preds,masks)
                pix_acc, mIoU = score.get(preds,masks)
                log.custom_scalar("PixelAccuracy/validation", pix_acc, epoch)
                log.custom_scalar("ScoreMIOU/validation", mIoU, epoch)
                log.val_iou_per_class(iou_dict, epoch)
                log.val_miou(iou_mean,epoch)
                log.display_val_batch(images, masks, Cityscapes.decode_labels(preds,colors), epoch, unNorm = True)
                

                if np.mean(valid_loss) < best_validation_loss:
                    best_validation_loss = np.mean(valid_loss)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': best_validation_loss},
                                'weights/FIbNet_CityScape.pt')