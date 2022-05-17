
from . utils import *


def reset_window(img,ww=1800,wl=-500,RescaleSlope=1.0,RescaleIntercept = -1024,if_dicom=True):
    if if_dicom:
        img     = img*int(RescaleSlope) + RescaleIntercept
    minWindow   = wl - ww*0.5
    img         = (img-minWindow)/ww
    img[img>1]  = 1
    img[img<0]  = 0
    return img