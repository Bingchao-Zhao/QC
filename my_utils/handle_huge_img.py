import os
import copy
# from multiprocessing import Manager,Process,Lock
# import math
# import threading

import cv2
import numpy as np
from openslide import OpenSlide
import pyvips

from . utils import *
from . import handle_img as hi
from . import file_util as fu

SMALL_PATCH = 64*224
MAX_SIZE = 150000
COMPRESS_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 80]
INDEX_SPLIT = '#'
NAME = "{}".format(INDEX_SPLIT)+"{}"+\
        "{}".format(INDEX_SPLIT)+"{}"+\
        "{}".format(INDEX_SPLIT)+"{}"+\
        "{}".format(INDEX_SPLIT)+"{}"+'.jpg'

#get file by index
gfi = lambda img,ind : copy.deepcopy(img[ind[0]:ind[1], ind[2]:ind[3]])

def gen_patches_index(ori_size, *, img_size=224, stride = 224,keep_last_size = False):

    """
        这个函数用来按照输入的size和patch大小，生成每个patch所在原始的size上的位置

        keep_last_size：表示当size不能整除patch的size的时候，最后一个patch要不要保持输入的img_size
        
        返回：
            一个np数组，每个成员表示当前patch所在的x和y的起点和终点如：
                [[x_begin,x_end,y_begin,y_end],...]
    """
    height, width = ori_size[:2]
    index = []
    if height<img_size or width<img_size: 
        warn("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        return index
        
    for h in range(0, height+1, stride):
        xe = h+img_size
        if h+img_size>height:
            xe = height
            h = xe-img_size if keep_last_size else h

        for w in range(0, width+1, stride):
            ye = w+img_size
            if w+img_size>width:
                ye = width
                w = ye-img_size if keep_last_size else w
            index.append(np.array([h, xe, w, ye]))

            if ye==width:
                break
        if xe==height:
            break
    return index


def get_patches(img,*, img_size=224, stride = 224, append_rate = 0.5,\
                keep_last_size = True, mask=[],return_patch=True):
    """
        这个函数用来按照patch的大小将整个img切割成patch
        keep_last_size：表示当size不能整除patch的size的时候，最后一个patch要不要保持输入的img_size
        
        返回：
            patch   : [patch1,patch2]
            index   : [[x_begin,x_end,y_begin,y_end],...]
    """
    height, width = img.shape[:2]
    if height<img_size or width<img_size: 
        warn("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        if return_patch:
            return [],[]
        return []

    if np.shape(mask)[0] != 0:
        mask = mask>0
        shape_mask = mask.shape[0:2]
        shape_img = mask.shape[0:2]
        if shape_mask != shape_img:
            err("shape_mask {}donot equal to shape_img {}".format(shape_mask,shape_img))
    # print(np.sum(mask))
    index = gen_patches_index(img.shape[:2],img_size=img_size,\
                             stride = stride,keep_last_size = keep_last_size)
    patches, ret_index = [], []

    for ind in index:
        _img = gfi(img, ind)
        if np.shape(mask)[0] != 0:
            # print(np.sum(mask[ind[0]:ind[1], ind[2]:ind[3]]),img_size*img_size*append_rate)
            if np.sum(gfi(mask, ind))<img_size*img_size*append_rate:
                continue
        if return_patch:
            patches.append(_img)
        ret_index.append(ind)
        
    if return_patch:
        return patches, ret_index
    return ret_index

#废弃的函数
def create_patch_index(patch,size=224,stride=224):
    size, stride    = int(size), int(stride)
    new_p           = np.array(patch)
    height, width   = new_p.shape[0], new_p.shape[1]

    if height < size or width < size:
        err("Patch size ({}-{}) Less than size({})".format(height, width,size))
        return None, None
    ret_x = []
    ret_y = []
    for i in range(0,99999):
        s1 = i*stride if (i*stride + size) <= height else height-size
        for j in range(0,99999):
            s2 = j*stride if (j*stride + size) <= width else width-size
            ret_x.append([s1,s1+size])
            ret_y.append([s2,s2+size])

            if s2+size >= width: break
            
        if s2+size >= width and s1+size >= height:break
            
    return ret_x, ret_y
    
def find_revolution(level,reduction_ratio = 1):
    level_0 = level[0]
    for l,i in enumerate(level) :
        if round(level_0[0]/i[0]) == reduction_ratio:
            return l,i
    return None,None   

def get_WSI_image_of_level(filePath,scale = 1, small_size = SMALL_PATCH,\
                fast_read = False, defi_print = rtime_print, handle_fun = None,\
                handle_level = 1,    handle_info = [], return_img = True):
    """
        将完整的WSI，分成patch读取
        scale           : 表示返回的img,需要缩放的倍率,默认不缩放
        small_size      : 每次读取的patch的最大值
        fast_read       : 在svs能够直接读取相应倍率的图像是，是否直接读取，而不读取最大放大倍率
        handle_fun      : 自定义处理每个patch的函数，如果定义了
        handle_level    : 表示handle在哪个level执行，1表示读取的原分辨率，大于1表示在返回的分辨率
    """
    if not just_ff(filePath, file=True):
        err("'{}' does not exist!!".format(filePath))
    if scale != 1:
        tips(" The reduction_ratio of WSI is {} ".format(scale))

    try:
        slide   = OpenSlide(filePath)
    except:
        err("Read '{}' error.".format(filePath),exit = False)
        return np.array([])

    # 小于最大倍率的图像，读取的时候很容易出现部分的黑色，这可能是由于解压时候错误导致的，目前原因不明
    if fast_read and os.path.splitext(filePath)[1] =='.svs' and scale != 1:
        level, size = find_revolution(slide.level_dimensions, scale)
        if level == None:
            tips("Donot find the level with scale = '{}' in file '{}'".\
                            format(scale,os.path.basename(filePath)))
        else:   
            return np.array(slide.read_region((0,0),level,size),dtype = np.uint8)[:,:,0:3]

    info('Image size:({})'.format(slide.level_dimensions[0]))
    size1,size2    = slide.level_dimensions[0][:2]
    # 这里只是想要这么大的内存，并不会真的马上占用所申请的内存，只有赋值之后才会被占用
    # 所有申请多大都不会导致内存不足
    if return_img:
        ret     = np.zeros([size2,size1,3],dtype = np.uint8)
    index   = gen_patches_index((size1,size2), img_size=small_size,stride=small_size,\
                                keep_last_size=True)

    for ind in index:
        defi_print("Decompression {}-{},{}-{} in {}-{}".\
                                format(ind[0],ind[1],ind[2],ind[3],size1,size2)+' '*10)
        patch = np.array(slide.read_region((ind[0],ind[2]), 0, (ind[1]-ind[0], ind[3]-ind[2])),\
                                dtype = np.uint8)[:,:,0:3]
        if handle_level <= 1 and handle_fun != None:
            handle_fun(patch, [ind[2],ind[3], ind[0],ind[1]], handle_info)

        new_ind = ind
        if scale != 1:
            new_ind = [i//scale for i in ind]
            patch   = cv2.resize(patch,(new_ind[1]-new_ind[0],new_ind[3]-new_ind[2]))

        if handle_level <= 1 and handle_fun != None:
            handle_fun(patch, [new_ind[2],new_ind[3], new_ind[0],new_ind[1]], handle_info)

        if return_img:
            ret[new_ind[2]:new_ind[3], new_ind[0]:new_ind[1],:] = patch

        del patch
        free_memory(if_cuda=False)

    if return_img:
        max_index = np.max(index,0)
        return ret[0:max_index[3]//scale, 0:max_index[1]//scale]

def pyvips_crop(svs_file, ind):
    img = pyvips.Image.new_from_file(svs_file)
    img2 = img.crop(ind[2], ind[0], ind[3]-ind[2], ind[1]-ind[0])
    img2 = np.asarray(img2, dtype=np.uint8)
    return img2

def creat_mutil_resolution_dir(svs_file, 
                                input_root, 
                                save_root, 
                                resolution=['40X', '20X', '10X']):
    floder = os.path.splitext(os.path.split(svs_file)[-1])[0]
    dir_name = fu.split_path(input_root, input_path=os.path.split(svs_file)[0])
    for reso in resolution:
        save_path_nX = os.path.join(save_root, reso, dir_name, floder)
        just_ff(save_path_nX, create_floder=True)

def just_mutil_resolution_patch(svs_file, 
                                input_root, 
                                save_root, 
                                ind,
                                name = NAME, 
                                resolution=['40X', '20X', '10X'], 
                                down_sample={'40X':1, '20X':2, '10X':4}):
    floder = os.path.splitext(os.path.split(svs_file)[-1])[0]
    dir_name = fu.split_path(input_root, input_path=os.path.split(svs_file)[0])
    exist = 0
    ret_name = []
    for reso in resolution:
        save_path_nX = os.path.join(save_root, reso, dir_name, floder)
        _ind = [int(i//down_sample[reso]) for i in ind]
        name_nX = os.path.join(save_path_nX,floder+name.\
                        format(_ind[0],_ind[1],_ind[2],_ind[3]))
        ret_name.append(name_nX)
        if just_ff(name_nX, file=True):
            exist += 1 

    return exist == len(resolution), ret_name

def write_mutil_patch(patch, name, 
                        resolution=['40X', '20X', '10X'], 
                        down_sample={'40X':1, '20X':2, '10X':4},
                        img_compress=COMPRESS_PARAMS):
    h,w = patch.shape[0:2]
    for num, n in enumerate(name):
        if just_ff(n, file=True):
            continue
        _h = int(h//down_sample[resolution[num]])
        _w = int(w//down_sample[resolution[num]])
        # some patch small than 4
        if _h*_w == 0:
            continue
        _patch = cv2.resize(patch, (_w, _h))
        hi.cv2_writer(file_path=n, img=_patch)

def crop_and_save_wsi(svs_file, 
                        input_root, 
                        save_root, 
                        name = NAME, 
                        just_hanle=False,
                        patch_size = SMALL_PATCH, 
                        img_compress=COMPRESS_PARAMS, 
                        resolution=['40X', '20X', '10X'], 
                        down_sample={'40X':1, '20X':2, '10X':4}):
    """_summary_

    Args:
        svs_file (_type_): _description_
        input_root (_type_): _description_
        save_root (_type_): _description_
        name (_type_, optional): _description_. Defaults to NAME.
        just_hanle (bool, optional): _description_. Defaults to False.
        patch_size (_type_, optional): _description_. Defaults to SMALL_PATCH.
        img_compress (_type_, optional): _description_. Defaults to COMPRESS_PARAMS.
        resolution (list, optional): _description_. Defaults to ['40X', '20X', '10X'].
        down_sample (dict, optional): _description_. Defaults to {'40X':1, '20X':2, '10X':4}.

    Returns:
        _type_: _description_
    """
    if not just_ff(svs_file, file=True):
        err("File '{}' is not exist!!".format(svs_file))
    if not just_hanle:
        tips("Begin to handle:'{}'".format(svs_file))

    img = pyvips.Image.new_from_file(svs_file)
    height, width = img.height, img.width
    index_list = gen_patches_index((height, width),
                                        img_size=patch_size, 
                                        stride=patch_size,
                                        keep_last_size=False)
    if len(index_list)==0:
        index_list=[[0,height, 0,width]]
    
    floder = os.path.splitext(os.path.split(svs_file)[-1])[0]
    if not just_hanle:
        creat_mutil_resolution_dir(svs_file, input_root, save_root, resolution=resolution)


    start = time.time()
    # hhi.get_WSI_image_of_level(svs_file, small_size=SIZE_BIG_PATCH, 
    #                     handle_fun=handle_patch, return_img=False)
    # print('using time:{}'.format(time.time()-start))
    handle = False
    for num, ind in enumerate(index_list) :
        if_crop, patch_name = just_mutil_resolution_patch(svs_file, 
                                                    input_root, 
                                                    save_root, 
                                                    ind, name = name,
                                                    resolution=resolution, 
                                                    down_sample=down_sample)
        if if_crop:
            continue

        if not just_hanle:
            rtime_print("loading index:'{}',num:'{}/{}' img:'{}'".\
                        format(ind, num, len(index_list), floder))
            img2 = pyvips_crop(svs_file, ind)
            write_mutil_patch(img2, patch_name, 
                                img_compress = img_compress,
                                resolution=resolution, 
                                down_sample=down_sample)
        handle = True
    if not just_hanle:
        info('Using time:{}'.format(time.time()-start))
    return handle

def order_image_name(img_path, suffix = 'png'):
    id_dict = {}
    ret = []
    files = os.listdir(img_path)
    for f in files:
        id = f.split("-")[0]
        if id_dict.get(id)==None:
            id_dict[id] = []
        id_dict[id].append(f)

    def order_img_index(images_path, suffix = "png"):
        ret = []
        id = images_path[0].split("-")[0]
        for i in range(1, 10000):
            temp = []
            for j in range(1, 10000):
                file_name = "{}-{}-{}.{}".format(id,i, j,suffix)
                # print(file_name)
                if file_name not in images_path:
                    break
                temp.append(file_name)
            if temp==[]: break
            ret.append(temp)
        return ret

    for key, value in id_dict.items():
        order_images = order_img_index(value,suffix)
        ret.append(order_images)
    return ret



def read_wsi_patch(file_patch, splite = '#'):
    f = os.path.split(file_patch)[-1]
    id_, x_b, x_e, y_b, y_e = f.split(splite)[0:5]
    y_e = y_e.split('.')[0]
    x_b, x_e, y_b, y_e = [int(i) for i in [x_b, x_e, y_b, y_e]]
    img = hi.cv2_reader(file_patch)
    return [x_b, x_e, y_b, y_e], img

def patch_worker(member, patches_dic, num, lock, splite = '#'):
    for file_path in member:
        ind, img = read_wsi_patch(file_path, splite)
        if  lock.acquire():
            patches_dic[str(ind)] = [ind, img]
            lock.release() 
        




def reconstruct_wsi(files_floder,splite = '#', processer=10):
    """
        将小块的patch重组成完整的patch，patch的格式为：
        id-x_begin-x_end-y_begin-y_end.png
    """
    ret  = np.zeros([MAX_SIZE,MAX_SIZE,3],dtype = np.uint8)
    files = os.listdir(files_floder)
    H,W = 0,0
    if processer>1:
        patches_dic = {}
        mp.mutil_thread([os.path.join(files_floder, f) for f in files], 
                        patch_worker, processer, 
                        patches_dic=patches_dic, splite='#')

        if len(patches_dic.keys()) != len(files):
            err('mutil processor num:{} donot equal to file num:{}'.\
                format(len(patches_dic.keys()),len(files)))
        
        for key, [ind, img] in patches_dic.items():
            ret[ind[0]:ind[1], ind[2]:ind[3]] = img
            H = ind[1] if ind[1] > H else H
            W = ind[3] if ind[3] > W else W
        patches_dic.clear()
        return ret[0:H, 0:W]
    else:
        for f in files:
            ind, img = read_wsi_patch(os.path.join(files_floder, f), splite=splite)
            ret[ind[0]:ind[1], ind[2]:ind[3]] = img
            H = ind[1] if ind[1] > H else H
            W = ind[3] if ind[3] > W else W

    return ret[0:H, 0:W]
    

def image_with_same_name(files, suffix = ".png"):
    id_dict = {}
    for f in files:
        _suffix = os.path.splitext(f)[-1]
        if _suffix != suffix:
            continue
        id = f.split("-")[0]
        if id_dict.get(id)==None:
            id_dict[id] = []
        id_dict[id].append(f)
    return id_dict

def Splice_patches_by_index(patches, index, channel_first=True, overlay_model='max'):
    """Splicing the patches by index.

    Args:
        img (list): a list of img patches, must have the same number of index.
        index (list): a list of patches index. [[indx_begin, indx_end, indy_begin, indy_end]]
        channel_first (bool, optional): . Defaults to True.
        overlay_model (str, optional): max or add. Defaults to max.

    Returns:
        numpy.ndarray: a recontrure numpy image.
    """
    assert len(patches)==len(index), \
        'len(img)={}, mismatch with len(index)={}'.format(len(patches), len(index))
    width, height = 0, 0
    for _, indx, _, indy in index:
        width = indx if indx>width else width
        height = indy if indy>height else height
    
    ret = None
    if len(patches[0]) == 2:
        ret = np.zeros((width, height))
    elif len(patches[0]) == 3 and channel_first:
        ret = np.zeros((patches[0].shape[0], width, height))
    elif len(patches[0]) == 3 and not channel_first:
        ret = np.zeros((width, height, patches[0].shape[0]))
    else:
        err("Size patches[0] is not right! (patches[0].shape={})".format(patches[0].shape))
    
    if overlay_model=='max':
        overlay_oper = np.add
    elif overlay_model=='add':
        overlay_oper = np.maximum
    else:
        err("overlay_model has a error value '{}'".format(overlay_model))
    

    for num, ind in enumerate(index):
        if channel_first:
            ret[..., ind[0]:ind[1], ind[2]:ind[3]] = \
                overlay_oper(patches[num], ret[..., ind[0]:ind[1], ind[2]:ind[3]])
        else:
            ret[ind[0]:ind[1], ind[2]:ind[3], ...] = \
                overlay_oper(patches[num], ret[ind[0]:ind[1], ind[2]:ind[3], ...])
    return ret


def read_wsi_with_img(img_path, p_splite='#'):
    if os.path.isfile(img_path):
        img = hi.img_reader(img_path)
    elif os.path.isdir(img_path):
        img = reconstruct_wsi(img_path, splite=p_splite)
    else:
        err('ERROR file type:"{}"'.format(img_path))
    
    return img
