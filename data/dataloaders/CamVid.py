import argparse
import os
from os.path import join as join
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
from utils import PILtoLongTensor
from collections import OrderedDict
import torchvision

class Generator(torch.utils.data.Dataset):
    def __init__(self, root_dir = '/home/spacelab/Documents/CamVid/', dataset = 'train', transforms = ""):

        self.transforms = transforms
        self.root_dir   = root_dir
        self.dataset    = dataset
        self.image_path_of = {
                            'train':join(self.root_dir, 'train'),
                            'val':join(self.root_dir, 'val'),
                            'test':join(self.root_dir, 'test')
                            }

        self.mask_path_of  = {
                            'train':join(self.root_dir, 'train_labels'),
                            'val':join(self.root_dir, 'val_labels'),
                            'test':join(self.root_dir, 'test_labels')
                             }
        self.encoded_mask_path_of  = {
                                    'train':join(self.root_dir, 'encoded_labels/train'),
                                    'val':join(self.root_dir, 'encoded_labels/val'),
                                    'test':join(self.root_dir, 'encoded_labels/test')
                                        }  
        self.image_set  = {
                            'train': os.listdir(self.image_path_of['train']),
                            'val': os.listdir(self.image_path_of['val']),
                            'test': os.listdir(self.image_path_of['test'])
                            }
        self.mask_set   = {
                            'train': os.listdir(self.mask_path_of['train']),
                            'val': os.listdir(self.mask_path_of['val']),
                            'test': os.listdir(self.mask_path_of['test'])
                            }
        if [fn for fn in self.mask_set[self.dataset]] != os.listdir(self.encoded_mask_path_of[self.dataset]):
            self.encode_class(self.dataset)
    def __len__(self):
        return len(self.image_set[self.dataset])
    
    def __getitem__(self, idx):
        image = Image.open(join(self.image_path_of[self.dataset],self.image_set[self.dataset][idx]))
        mask  = Image.open(join(self.mask_path_of[self.dataset], self.image_set[self.dataset][idx][:-4]+'_L.png'))
        encoded_mask  = Image.open(join(self.encoded_mask_path_of[self.dataset], self.image_set[self.dataset][idx][:-4]+'_L.png'))

        image, encoded_mask = self.transform(image, encoded_mask, self.transforms)
        return image, encoded_mask, self.pil_to_long_tensor(mask) 

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
            if random.random() > 0.5:
                brightness = random.uniform(1.2, 1.8)
                image = TF.adjust_brightness(image,brightness)
        if('contrast' in self.transforms):
            if random.random() > 0.5:
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
            image = TF.resize(image, (480,480))
            mask = TF.resize(mask, (480,480))
        if('fcrop' in self.transforms):
            image = TF.five_crop(image,(224,224))
            mask = TF.five_crop(mask, (224,224))
        if('center_crop' in self.transforms):
            image = TF.center_crop(image, (480,480))
            mask = TF.center_crop(mask, (480,480))
       
        mask = self.pil_to_long_tensor(mask)
        image = TF.to_tensor(image)
        
        if('normalize' in self.transforms):
                    image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return image, mask



    def pil_to_long_tensor(self, mask):

        mask_byte = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))

        n_channels = len(mask.mode)

        mask = mask_byte.view(mask.size[1], mask.size[0], n_channels)

        return mask.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()

    def encode_class(self, opt = 'train'):
        print("Encoding masks for ", opt)
        for  i , fn in enumerate(tqdm(self.mask_set[opt], total = len(self.mask_set[opt]), leave = True)):
            mask = cv.imread(join(self.mask_path_of[opt],fn))
            mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
            mask = mask.astype(int)
            encoded_mask = np.zeros((mask.shape[0],mask.shape[1]), dtype = np.int16)
            for encoding_idx, label in enumerate(self.CamVid_labels_array()):
                encoded_mask[np.where(np.all(mask == self.CamVid_labels_array()[encoding_idx], axis = -1))[:2]] = encoding_idx
            encoded_mask = encoded_mask.astype(int)
            cv.imwrite(join(self.encoded_mask_path_of[opt], fn),encoded_mask)
    
    def decode_mask(self, mask):

        CamVid_labels_array = self.CamVid_labels_array()
        mask = mask.to('cpu')
        mask = torch.argmax(mask, dim=1)
        pred_img = [CamVid_labels_array[px] for px in mask] 
        image_grid = torch.tensor(pred_img)
        image_grid = image_grid.permute(0,3,1,2)
        return image_grid

    def labels_of_interest(self):
        return np.array(['Void','Building', 'Tree', 'Sky', 'Car', 'Sign', 'Road', 'Pedestrian', 'Fence', 'Pole', 'Sidewalk', 'Cyclist'])

    def CamVid_labels_array(self):
        return np.array([
          [0, 0, 0],
          [128, 0, 0],
          [128, 128, 0],
          [128, 128, 128],
          [64, 0, 128],
          [192, 128, 128],
          [128, 64, 128],
          [64, 64, 0],
          [64, 64, 128],
          [192, 192, 128],
          [0, 0, 192],
          [0,128, 192]
            ])

    def CamVid_labels(self):
        return OrderedDict([
            ('Animal', [64,128,64]),
            ('Archway', [192,0,128]),
            ('Cyclist', [0,128, 192]),
            ('Bridge', [0, 128, 64]),
            ('Building', [128, 0, 0]),
            ('Car', [64, 0, 128]),
            ('CartLuggagePram', [64, 0, 192]),
            ('Child', [192, 128, 64]),
            ('Pole', [192, 192, 128]),
            ('Fence', [64, 64, 128]),
            ('LaneMkgsDriv', [128, 0, 192]),
            ('LaneMkgsNonDriv', [192, 0, 64]),
            ('Misc_Text', [128, 128, 64]),
            ('MotorcycleScooter', [192, 0, 192]),
            ('OtherMoving', [128, 64, 64]),
            ('ParkingBlock', [64, 192, 128]),
            ('Pedestrian', [64, 64, 0]),
            ('Road', [128, 64, 128]),
            ('RoadShoulder', [128, 128, 192]),
            ('Sidewalk', [0, 0, 192]),
            ('Sign', [192, 128, 128]),
            ('Sky', [128, 128, 128]),
            ('SUVPickupTruck', [64, 128,192]),
            ('TrafficCone', [0, 0, 64]),
            ('TrafficLight', [0, 64, 64]),
            ('Train', [192, 64, 128]),
            ('Tree', [128, 128, 0]),
            ('Truck_Bus', [192, 128, 192]),
            ('Tunnel', [64, 0, 64]),
            ('VegetationMisc', [192, 192, 0]),
            ('Void', [0, 0, 0]),
            ('Wall', [64, 192, 0]),
            ])