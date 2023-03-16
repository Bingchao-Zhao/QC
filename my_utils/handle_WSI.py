import cv2
import numpy as np
from . import handle_img as hi
import skimage
import skimage.morphology
from .utils import *

def remove_small_hole(mask, h_size=10):
    """remove the small hole

    Args:
        mask (_type_): a binary mask, can be 0-1 or 0-255
        h_size (int, optional): min_size of the hole

    Returns:
        mask
    """
    value = np.unique(mask)
    if len(value)>2:
        err(f"Input mask should be a binary, but get value:({value})")
    pre_mask_rever = mask==0
    pre_mask_rever = skimage.morphology.remove_small_objects(pre_mask_rever, \
                                                            min_size=h_size)
    mask[pre_mask_rever<=0] = 1
    return mask

def ostu_seg_tissue(img, 
                    remove_hole:bool=True, 
                    min_size:int=4000,
                    mod:str='cutoff'):
    """Segmentation tissue of WSI with ostu.

    Args:
        img (_type_): Input img. Must in 10X resolution.
        remove_hole (bool, optional): remove small hole or not
        min_size (int, optional): min_size of hole

    Returns:
        _type_: _description_
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise using a Gaussian filter
    gray = cv2.GaussianBlur(gray, (35,35), 0)
    # Otsu thresholding and mask generation
    if mod=="cutoff":
        thresh_otsu = gray < 234
    elif mod=="ostu":
        ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if remove_hole:
        thresh_otsu = remove_small_hole(thresh_otsu, min_size)
        thresh_otsu = thresh_otsu!=0
        thresh_otsu = remove_small_hole(thresh_otsu, min_size)
    else:
        thresh_otsu = thresh_otsu!=0
    return thresh_otsu
