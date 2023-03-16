import math
import copy

import cv2
import numpy as np
import torch
import skimage
from torchvision.utils import make_grid
import skimage.measure
import skimage.morphology
from PIL import Image
import matplotlib.pyplot as plt

from .utils import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def pp(img,dpi = 100):
    plt.figure(dpi)
    plt.imshow(img)

def img_reader(file_path):
    img = Image.open(file_path)
    return np.array(img, dtype=np.uint8)

def cv2_reader(file_path):
    img = cv2.imread(file_path)
    if len(img.shape) >=3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2_writer(file_path, img, *,color='BGR'):
    if type(img) is not np.ndarray:
        img = np.array(img, dtype = np.uint8)
    if len(img.shape) >=3 and color=="BGR":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_path, img)

def cv2_resize(file, magnigicine = 1, interpolation=cv2.INTER_NEAREST, *args):
    size1, size2 = np.shape(file)[1], np.shape(file)[0]
    size1, size2 = int(size1*magnigicine), int(size2*magnigicine)
    return cv2.resize(file, (size1, size2), interpolation,*args)

def cat_chann_f_img(img, row_num = 1, channel_frist = False):
    max_channel = 3
    max_dim = 1
    for i in img:
        if len(i.shape)!=3 and len(i.shape)!=2:
            err("Only accept 2d or 3d tensor!")
        if len(i.shape)==3 and (i.shape[0]!=1 and i.shape[0]!=3):
            err("Olny accept 1 or 3 channel img")
    
def concat_img(images, channel_frist=True, nrow=1, padding=10):
    temp_images = copy.deepcopy(images)
    if isinstance(images, list):
        if np.sum([(len(img.shape)!=2 and len(img.shape)!=3) for img in temp_images])>0:
            err("Have one or more images isnot 2d or 3d")
        
        m_w, m_h = 0, 0
        three_d = False
        data_type = torch.uint8 if temp_images[0].max() >1 else torch.float32
        mean_value = 128 if temp_images[0].max() >1 else 0.5
        for img in temp_images:
            if len(img.shape) ==3:
                three_d = True

        for i in range(len(temp_images)):
            if type(temp_images[i]) is np.ndarray:
                temp_images[i] = torch.from_numpy(temp_images[i])

            if not channel_frist and len(temp_images[i].shape) == 3:
                temp_images[i] = temp_images[i].permute(2,0,1)
            elif three_d and len(temp_images[i].shape) == 2:
                temp_images[i] = torch.stack([temp_images[i], temp_images[i], temp_images[i]], 0)
            elif len(temp_images[i].shape) == 2:
                temp_images[i] = temp_images[i].unsqueeze(0)
        
        for img in temp_images:
            if three_d:
                m_w = m_w if img.shape[1]<m_w else img.shape[1]
                m_h = m_h if img.shape[2]<m_h else img.shape[2]
            else:
                m_w = m_w if img.shape[0]<m_w else img.shape[0]
                m_h = m_h if img.shape[1]<m_h else img.shape[1]

        for i in range(len(temp_images)):    
            if temp_images[i].shape[1] < m_w or temp_images[i].shape[2] < m_h:
                _, t_w, t_h = temp_images[i].shape
                new_img = torch.ones((temp_images[i].shape[0], m_w, m_h), \
                                        dtype=data_type)*mean_value
                new_img[:,  (m_w-t_w)//2 : (m_w-t_w)//2+t_w, \
                            (m_h-t_h)//2 : (m_h-t_h)//2+t_h] = temp_images[i]
                temp_images[i] = new_img
        temp_images = torch.stack(temp_images)
        
    elif type(temp_images) is np.ndarray:
        if len(temp_images.shape) != 3 and len(temp_images.shape) != 4:
            err('Error shape of input:{}'.format(temp_images.shape))

        temp_images = torch.from_numpy(temp_images)
        if len(temp_images.shape) ==3:
            temp_images = temp_images.unsqueeze(1)

        if not channel_frist:
            temp_images = temp_images.permute(0, 3, 1, 2)
    elif type(temp_images) is torch.Tensor:
        if len(temp_images.shape) != 3 and len(temp_images.shape) != 4:
            err('Error shape of input:{}'.format(temp_images.shape))

        if len(temp_images.shape) ==3:
            temp_images = temp_images.unsqueeze(1)

        if not channel_frist:
            temp_images = temp_images.permute(0, 3, 1, 2)

    ret = make_grid(temp_images, nrow=math.ceil(len(temp_images)/nrow), padding=padding)
    if not channel_frist:
        return ret.permute(1,2,0)
    return ret
def get_color_mask(inst_map):
    """color the instance with random color

    Args:
        inst_map (numpy): can be a binary map or a instance map

    Returns:
        (H, W, 3): a instance color map
    """
    h, w = inst_map.shape[0:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    inst_map = skimage.measure.label(inst_map)
    inst_id = np.unique(inst_map)
    for ins in inst_id:
        if ins<=0: continue
        color_mask[inst_map==ins, 0] = random.randint(0, 255)
        color_mask[inst_map==ins, 1] = random.randint(0, 255)
        color_mask[inst_map==ins, 2] = random.randint(0, 255)
    return color_mask

def add_mask(img, mask, channel_frist=False):
    if len(img.shape)==2 or channel_frist:
        return img*mask
    if not channel_frist and torch.is_tensor(img):
        _temp = img.permute(2,0,1)*mask
        return  _temp.permute(1,2,0)
    if not channel_frist and type(img) is np.ndarray:
        _temp = img.transpose(2,0,1)*mask
        return  _temp.transpose(1,2,0)



