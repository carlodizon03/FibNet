from tensorboardX import SummaryWriter
from datetime import datetime 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
class Create(object):

    def __init__(self, path):
        self.timestamp = datetime.now()
        self.str_timestamp = str("{0}-{1}-{2}--{3}-{4}-{5}/".format(self.timestamp.month, self.timestamp.day,self.timestamp.year,self.timestamp.hour,self.timestamp.minute,self.timestamp.second))
        self.log_dir =  os.path.join(path, self.str_timestamp)
        self.writer = SummaryWriter(self.log_dir)

    def custom_scalar(self, path_name, value, step):
        self.writer.add_scalar(path_name,value,step)
        self.writer.flush()

    def train_loss_per_epoch(self, value, step):
        self.writer.add_scalar('Loss/training_per_epoch', value, step)
        self.writer.flush()

    def train_loss(self, value, step):
        self.writer.add_scalar('Loss/training', value, step)
        self.writer.flush()

    def train_mloss(self, value, step):
        self.writer.add_scalar('mLoss/training', value, step)
        self.writer.flush()
    
    def train_acc(self, value, step):
        self.writer.add_scalar('Accuracy/training', value, step)
        self.writer.flush()

    def train_top1_avg(self, value, step):
        self.writer.add_scalar('top1_avg/training', value, step)
        self.writer.flush()
    def train_top5_avg(self, value, step):
        self.writer.add_scalar('top5_avg/training', value, step)
        self.writer.flush()

    def train_top1(self, value, step):
        self.writer.add_scalar('top1/training', value, step)
        self.writer.flush()
    def train_top5(self, value, step):
        self.writer.add_scalar('top5/training', value, step)
        self.writer.flush()

    def train_macc(self, value, step):
        self.writer.add_scalar('mAccuracy/training', value, step)
        self.writer.flush()
    
    def train_DiceIoU(self,iou,step):
        self.writer.add_scalar('DiceIoU/training', iou, step)
        self.writer.flush()

    def train_DicemIoU(self,miou,step):
        self.writer.add_scalar('DicemIoU/training', miou, step)
        self.writer.flush()

    def train_mIoU(self,miou,step):
        self.writer.add_scalar('mIoU/training', miou, step)
        self.writer.flush()
        
    def train_iou_per_class(self,iou_dict,step):
        '''accepts dictionary of class:iou'''
        self.writer.add_scalars('IoU/training', iou_dict, step)
        self.writer.flush()

    def val_loss(self,value,step):
        self.writer.add_scalar('Loss/validation', value, step)
        self.writer.flush()
    
    def val_mloss(self,value,step):
        self.writer.add_scalar('mLoss/validation', value, step)
        self.writer.flush()
    
    def val_acc(self, value, step):
        self.writer.add_scalar('Accuracy/validation', value, step)
        self.writer.flush()
    
    def val_top1_avg(self, value, step):
        self.writer.add_scalar('top1_avg/validation', value, step)
        self.writer.flush()

    def val_top5_avg(self, value, step):
        self.writer.add_scalar('top5_avg/validation', value, step)
        self.writer.flush()

    def val_top1(self, value, step):
        self.writer.add_scalar('top1/validation', value, step)
        self.writer.flush()

    def val_top5(self, value, step):
        self.writer.add_scalar('top5/validation', value, step)
        self.writer.flush()

    def val_macc(self, value, step):
        self.writer.add_scalar('mAccuracy/validation', value, step)
    
        self.writer.flush()
    def val_loss_per_epoch(self, value, step):
        self.writer.add_scalar('Loss/validation_per_epoch', value, step)
        self.writer.flush()
        
    def val_dsc(self,value,step):
        self.writer.add_scalar('Loss/DSC_per_Volume',value,step)
        self.writer.flush()

    def val_DiceIoU(self,iou,step):
        self.writer.add_scalar('DiceIoU/validation', iou, step)
        self.writer.flush()

    def val_DicemIoU(self,miou,step):
        self.writer.add_scalar('DicemIoU/validation', miou, step)
        self.writer.flush()

    def val_miou(self,miou,step):
        self.writer.add_scalar('mIoU/validation',miou,step)
        self.writer.flush()

    def val_iou_per_class(self,iou_dict,step):
        '''accepts dictionary of class:iou'''
        self.writer.add_scalars('IoU/validation', iou_dict, step)
        self.writer.flush()

    def val_acc(self,value,step):
        self.writer.add_scalar('Accuracy/validation', value, step)
        self.writer.flush()

    def test_loss(self, value, step):
        self.writer.add_scalar('Loss/testing', value, step)
        self.writer.flush()

    def test_iou_per_class(self,iou_dict,step):
        '''accepts dictionary of class:iou'''
        self.writer.add_scalars('IoU/testing', iou_dict, step)
        self.writer.flush()
        
    def test_miou(self,miou,step):
        self.writer.add_scalar('mIoU/testing',miou,step)
        self.writer.flush()

    def display_test_batch(self,images,masks,preds,epoch,unNorm = False, num_images = 4):
        
        if unNorm:
            images = self.unnorm(images)
        images = images.to('cpu')
        masks = masks.to('cpu')
        preds = preds.to('cpu')
        image_grid  = torchvision.utils.make_grid(images[:num_images])
        mask_grid   = torchvision.utils.make_grid(masks[:num_images])
        pred_grid   = torchvision.utils.make_grid(preds[:num_images])
        self.writer.add_image('Testing/image',image_grid,epoch)
        self.writer.add_image('Testing/mask',mask_grid,epoch)
        self.writer.add_image('Testing/prediction',pred_grid,epoch)
        self.writer.flush()

    def display_val_batch(self,images,masks,preds,epoch,unNorm = False,num_images = 4):
        
        if unNorm:
            images = self.unnorm(images)
        images = images.to('cpu')
        masks = masks.to('cpu')
        preds = preds.to('cpu')
        image_grid  = torchvision.utils.make_grid(images[:num_images])
        mask_grid   = torchvision.utils.make_grid(masks[:num_images])
        pred_grid   = torchvision.utils.make_grid(preds[:num_images])
        self.writer.add_image('Validation/image',image_grid,epoch)
        self.writer.add_image('Validation/mask',mask_grid,epoch)
        self.writer.add_image('Validation/prediction',pred_grid,epoch)
        self.writer.flush()
    
    def display_train_batch(self,images,masks,preds,epoch,unNorm = False,num_images = 4):
        
        if unNorm:
            images = self.unnorm(images)
        images = images.to('cpu')
        masks = masks.to('cpu')
        image_grid  = torchvision.utils.make_grid(images[:num_images])
        mask_grid   = torchvision.utils.make_grid(masks[:num_images])
        preds_grid = torchvision.utils.make_grid(preds[:num_images])
      
        self.writer.add_image('Training/image',image_grid,epoch)
        self.writer.add_image('Training/mask',mask_grid,epoch)
        self.writer.add_image('Training/prediction',preds_grid,epoch)
        self.writer.flush()

    def display_custom_batch(self,masks, mask_preds, epoch, images = None, unNorm = False, num_images = 4, name = 'Custom'):
        if unNorm and images!=None:
            images = self.unnorm(images)
        if images != None:
            image_grid  = torchvision.utils.make_grid(images[:num_images])
            self.writer.add_image(name + '/image',image_grid,epoch)
        
        mask_grid   = torchvision.utils.make_grid(masks[:num_images])
        preds_grid = torchvision.utils.make_grid(mask_preds[:num_images])
        self.writer.add_image(name + '/mask',mask_grid,epoch)
        self.writer.add_image(name + '/prediction',preds_grid,epoch)
        self.writer.flush()

    def model_graph(self,model,input_image):
        self.writer.add_graph(model,input_image,verbose=False)
        self.writer.flush()
        self.writer.close()

    def close(self):
        self.writer.close()

    