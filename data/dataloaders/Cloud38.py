import argparse
import os
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
from utils import Display
from utils import PILtoLongTensor



class Generator(torch.utils.data.Dataset):

    def __init__(self, non_empty = True, root_dir = "/home/spacelab/Documents/38Cloud/",  is_train = 'train',transforms = None):
        
        self.transforms     = transforms
        if non_empty:
            self.root_dir   = os.path.join(root_dir, "non_empty")
        
        self.image_path     = os.path.join(self.root_dir,"img/")
        self.mask_path      = os.path.join(self.root_dir,"mask/")
        
        self.dataset       = {
                                'image': os.listdir(self.image_path),
                                'mask'  : os.listdir(self.mask_path)
                              }

        
            
    def splits(self, division = [80,20], add_excess_to = 'train', save_file_names_to = None):
        
        total_images = self.__len__()
        if division[0] < 1 and division[1] < 1:
            splits = [int(total_images*division[0]), int(total_images*division[1])]
        elif division[0] > 1 and division[1] > 1:
            splits =  [int(total_images*(division[0]/100)), int(total_images*(division[1]/100))]

        if sum(splits) < total_images:
                excess = total_images - sum(splits)
                if 'train' == add_excess_to:
                    splits[0] = splits[0]+excess
                elif 'val' == add_excess_to:
                    splits[1] = splits[1]+excess
                return splits
        if(save_file_names_to is not None):
            
        return splits   

    def __len__(self):
        
        assert len(os.listdir(self.mask_path)) == len(os.listdir(self.image_path)), "Number of images and masks are not equal!"
        return len(os.listdir(self.mask_path))

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.image_path, self.dataset['image'][idx]))
        mask  = Image.open(os.path.join(self.mask_path, self.dataset['mask'][idx]))
        
        if self.transforms != None:
            image, mask = self.transform(image, mask, self.transforms)
        else: 
            image = TF.to_tensor(image)
            mask  = TF.to_tensor(mask) 
        return image,  mask

    def transform(self, image, mask, transforms):
    
        if('hflip' in self.transforms):
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        if('vflip' in self.transforms):
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        if('rotate' in self.transforms):
            if random.random() > 0.5:
                angle = random.randint(-15,15)
                image = TF.rotate(image,angle)
                mask = TF.rotate(mask,angle)    
        if('brightness' in self.transforms):
            if random.random() > 0.2:
                brightness = random.uniform(1.7, 1.9)
                image = TF.adjust_brightness(image,brightness)
        if('contrast' in self.transforms):
            if random.random() > 0.6:
                contrast = random.uniform(1.2,1.5)
                image = TF.adjust_contrast(image,contrast)
        if('hue' in self.transforms):
            if random.random() > 0.5:
                hue = random.uniform(0.1,0.3)
                image = TF.adjust_hue(image,hue)
        if('saturation' in self.transforms):
            if random.random() > 0.5:
                saturation = random.uniform(1.1,1.8)
                image = TF.adjust_saturation(image,saturation)
        if('adversarial' in self.transforms):
            if random.random() > 0.5:
                image = self.adjust_gamma(image,mask)
        if('resize' in self.transforms):
            image = TF.resize(image, (224,224))
            mask = TF.resize(mask, (224,224))
        if('fcrop' in self.transforms):
            image = TF.five_crop(image,(224,224))
            mask = TF.five_crop(mask, (224,224))
        if('center_crop' in self.transforms):
            image = TF.center_crop(image, (224,224))
            mask = TF.center_crop(mask, (224,224))
        
        mask = TF.to_tensor(mask)
        image = TF.to_tensor(image)
        if('normalize' in self.transforms):
                    image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    
        return image, mask
        
    def class_weighing(self, dataset = 'train', c = 1.02):
        print("Class weighing.")
        num_classes = 1
        class_count = 0
        total = 0
        for _, fn in enumerate(tqdm(self.dataset['mask'], total = len(self.dataset['mask']), leave = True)):
            mask = cv.imread(os.path.join(self.mask_path, fn))
            mask = np.array(mask/255).astype(np.int8)
            flat_mask = mask.flatten()
            class_count += np.bincount(flat_mask, minlength = num_classes)
            total += len(flat_mask)
        propensity_score = class_count / total
        class_weights = 1 / (np.log(c + propensity_score))
        return class_weights

    def pil_to_long_tensor(self, mask):

        mask_byte = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))

        n_channels = 1

        mask = mask_byte.view(mask.size[1], mask.size[0], n_channels)
        mask = mask/255.
        return mask.transpose(0, 1).transpose(0, 2).contiguous().long()

