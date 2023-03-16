from os import listdir
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as tm
from torchvision import transforms
from my_utils.utils import *
import my_utils.handle_huge_img as hhi
import my_utils.handle_img as hi
import  copy
import my_utils.file_util as fu
import my_utils.handle_WSI as hw
INPUT_PATH = '/media/zhaobingchao/hanchu0011/TMA/data/TMA_all_points'
SAVE_PATH = '/media/zhaobingchao/hanchu0011/TMA/data/mask/10_mask'
SAVE_COM_PATH = '/media/zhaobingchao/hanchu0011/TMA/data/mask/10_mask_concat'

def get_png_path():
    ret = []
    png_list = find_file(INPUT_PATH,3, suffix='.png')
    for p_p in png_list:
        dir,n = os.path.split(p_p)
        ret.append([p_p, dir, n])
    return ret

def get_img(img_path):
    img = hi.cv2_reader(img_path)
    return img

def pipline(model, png_list):

    for png_p, _dir, p_name in tqdm(png_list) :
        split_path = fu.split_path(INPUT_PATH, _dir)
        save_mask_path = os.path.join(SAVE_PATH, split_path, p_name)
        save_concat_mask_path = os.path.join(SAVE_COM_PATH, split_path, p_name)
        if just_ff(save_mask_path,file=True) and just_ff(save_concat_mask_path,file=True):
            continue
        img = get_img(png_p)#[::2,::2,:]
        
        img2 = copy.deepcopy(img)
        img = np.pad(img,((224, 224), (224, 224), (0, 0)),mode='reflect')
        mask = gen_mask(model, img)
        ostu_mask = hw.ostu_seg_tissue(img)
        mask = mask*(ostu_mask>0)
        mask = mask[224:mask.shape[0]-224, 224:mask.shape[1]-224]
        img2[mask<=0] = 0
        mask_concat = hi.concat_img([img, img2],channel_frist=False)
        just_dir_of_file(save_mask_path)
        just_dir_of_file(save_concat_mask_path)
        hi.cv2_writer(save_mask_path, mask)
        hi.cv2_writer(save_concat_mask_path, mask_concat)

def gen_mask(model, img):
    img_size=224
    stride=56
    batchsize = 512
    indx_list = hhi.gen_patches_index(img.shape[0:2], img_size=img_size, \
                                                stride=stride, keep_last_size=True)
    indx_zip = MyZip(indx_list, batch=batchsize)    
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    transform = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        for num, ind in enumerate(indx_zip):
            # rtime_print("Processing {}/{}".format(num, len(indx_zip)))
            inds = ind[0]
            input_img_list = []
            for _ind in inds:
                input_img_list.append(transform(hhi.gfi(img, _ind)))
            input_img_list = torch.stack(input_img_list).cuda()
            pred = model(input_img_list).cpu().detach().numpy()
            for num, _ind in enumerate(inds):
                if pred[num][0] > pred[num][1]:
                    mask[_ind[0]:_ind[1], _ind[2]:_ind[3]] += 0 
                else:
                    mask[_ind[0]:_ind[1], _ind[2]:_ind[3]] += 1 
    mask = np.array((mask>4)*255, dtype=np.uint8)
    return mask

model       = tm.resnet34(pretrained=True)
model.fc    = nn.Linear(512 , 2)
model.cuda().eval()
pretrained_model = torch.load('weight/resnet34/epoch-35.pth') # 导入预训练权重
model.load_state_dict(pretrained_model) # 将与训练权重载入模型
png_list = get_png_path()
pipline(model, png_list)
