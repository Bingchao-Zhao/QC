# QC
Using Deep Learning for Pathology Image Quality Control

If you want to train a new network for quality control, you can use the train_resnet.py.
In the orignal code, our model classfy two class, tissue and blank (including blank and pollution).

If you only want to use our model to segmentate the tissue from WSI, you can use the seg_tissue.py. But our code donot process the 
WSI file (.svs or .tiff, etc.) directly. You have to save the WSI file to .png or .jpg, then use test_wsi.py to segmentate the tissue.

Finally, the checkpoint of the pretrained model in our data if availabel in:

链接：https://pan.baidu.com/s/1WZaM9TK8g9ooSbNBlXZWuw 
提取码：zkhb


## **Finally, please do not forget that the model pre-trained on our data can only handle images at 10X resolution.**


## Display of segmentation results.

![image](https://raw.githubusercontent.com/Bingchao-Zhao/QC/master/A-2.png)
