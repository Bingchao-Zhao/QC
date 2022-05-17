import os
import cv2
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision.transforms import transforms
logging.basicConfig(level=logging.DEBUG,format='%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
TRANSFORM = {"train" : transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply([transforms.RandomRotation(90)], p=.3),
                        transforms.RandomApply([transforms.ColorJitter(.1, .1, .1)], p=.3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                    (0.12062634, 0.1206659, 0.12071837))]),

            "test" :  transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                                (0.12062634, 0.1206659, 0.12071837))])
            }
class BasicDataset(Dataset):
    def __init__(self, imgs_dir,train=True,split = False):
        self.imgs_dir = imgs_dir
        rate = 0.2
        #print(listdir(imgs_dir))
        if train:
            self.ids = [file for file in listdir(imgs_dir)
                    if file.find('anno')<0 and file.find('train')>=0]
            if split:
                self.ids = self.ids[int(len(self.ids)*0.2):len(self.ids)]
        else:
            if split:
                self.ids = [file for file in listdir(imgs_dir)
                    if file.find('anno')<0 and file.find('train')>=0]
                self.ids = self.ids[0:int(len(self.ids)*0.2)]
            else:
                self.ids = [file for file in listdir(imgs_dir)
                        if file.find('anno')<0 and file.find('test')>=0]  
            #print(self.ids)    
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img = cv2.imread(os.path.join(self.imgs_dir,idx))
        #print(img.shape)
        extension = idx.split('.')[-1]
        img = np.transpose(img[...,::-1],(2,0,1))
        mask = None
        if os.path.isfile(os.path.join(self.imgs_dir,idx.split('.')[0]+'_anno.'+'png')):
            mask = cv2.imread(os.path.join(self.imgs_dir,idx.split('.')[0]+'_anno.'+'png'))>0.5
        else:
            mask = cv2.imread(os.path.join(self.imgs_dir,idx.split('.')[0]+'_anno.'+'bmp'))>0.5
        if np.max(img)>1:
            img = img/255.
        if len(mask.shape)>2:
            mask = mask[:,:,0]
        #print(mask.shape)
        return {
            'image': torch.FloatTensor(img.copy()),
            'mask': torch.unsqueeze(torch.FloatTensor(mask.copy()),0)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
