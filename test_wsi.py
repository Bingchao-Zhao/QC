from os import listdir
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as tm

from my_utils.utils import *
import my_utils.handle_huge_img as hhi
import my_utils.handle_img as hi
import my_utils.file_util as fu

def nor_big_patch(model, wsi_path):
    # wsi_path should be a image. .png .jpg so on.
    img_size=224
    stride=200
    batchsize = 128
    wsi = hi.img_read(wsi_path)
    indx_list = hhi.gen_patches_index(wsi.shape[0:2], img_size=img_size, \
                                                stride=stride, keep_last_size=True)
    indx_zip = MyZip(indx_list, batch=batchsize)    
    pred_img, record_img = [], []

    with torch.no_grad():
        for num, ind in enumerate(indx_zip):
            rtime_print("Processing {}/{}".format(num, len(indx_zip)))
            ind = ind[0]
            input_img_list = []
            for _ind in ind:
                input_img_list.append(torch.FloatTensor(hhi.gfi(wsi, _ind)))
            input_img_list = torch.stack(input_img_list).cuda().permute(0,3,1,2)
            pred = model(input_img_list).cpu().detach().numpy()
            pred_img.extend(pred)
            record_img.extend(np.ones(pred.shape))

    pred_img = hhi.Splice_patches_by_index(pred_img, indx_list, overlay_model='add')
    record_img = hhi.Splice_patches_by_index(record_img, indx_list, overlay_model='add')
    pred_img /= record_img
    pred_img[pred_img<0] = 0
    pred_img = pred_img.transpose(1,2,0)
    np.array(pred_img*255, np.uint8)

model       = tm.resnet34(pretrained=True)
model.fc    = nn.Linear(512 , 2)
model.cuda().eval()
pretrained_model = torch.load('weight/resnet34/epoch-10.pth') # 导入预训练权重
model.load_state_dict(pretrained_model, strict=False) # 将与训练权重载入模型

nor_big_patch(model, INPUT_PATH)