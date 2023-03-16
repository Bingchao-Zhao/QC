import torchvision
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os
import cv2
import argparse
import torchvision.models as tm
import my_utils.vis as vis
from random import shuffle
from my_utils.utils import *
import my_utils.handle_img as hi
import my_utils.handle_huge_img as hhi
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel,CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import Dataset
from sklearn.metrics import precision_score,recall_score,confusion_matrix,accuracy_score,f1_score, roc_auc_score
import Config
import pandas as pd
from sklearn.model_selection import train_test_split
import my_utils.read_wsi_roi as rwr
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',default=30,type=int)
parser.add_argument('--batch_size',default=256,type=int)
win = 0
num = 0
train_num= []
train_loss = []
best_f1 = 0
REVOLUTION = "20X"
MAX_PATCH_OF_WSI = 300
DATA_CENTER = "nanfang"
MODEL = 'resnet34'
LABEL_PATH  = "../../data/nanfang/train_and_text/train.csv"
conf = Config.config()

class ImageDataset(Dataset):
    DATA_ROOT = '/media/zhaobingchao/MyBook/glioma/data/tissue_seg'
    SIZE = 224
    TISSUE_CUTOFF = 235
    STRIDE = 150
    def __init__(self, n_class=2, type_ ='train',transforms = None):
        self.data_trian = []
        self.tissue = 0
        self.notissue = 0
        self.tissue_img = 0
        self.transforms = transforms
        self.n_class = n_class
        self.img_name_list = find_file(ImageDataset.DATA_ROOT, depth_down=2, suffix='.jpg')
        for img_name in self.img_name_list:
            if img_name.find('/blank/')>=0:
                self.data_trian.extend(self.load_blank(img_name))
            #"/media/zhaobingchao/MyBook/glioma/data/tissue_seg/blank"
            elif img_name.find('/tissue/')>=0:
                self.data_trian.extend(self.load_tissue(img_name))
            elif img_name.find('/pollution/')>=0:
                self.data_trian.extend(self.load_pollution(img_name))
        info("tissue:{}, notissue:{}".format(self.tissue, self.notissue))
    
    def load_blank(self, img_name):
        data = []
        img = hi.cv2_reader(img_name)
        # print(img)
        patch_ind_list = hhi.gen_patches_index(img.shape, 
                                            img_size=ImageDataset.SIZE, 
                                            stride=ImageDataset.STRIDE, 
                                            keep_last_size=True)
        # print(len(patch_ind_list))
        for ind in patch_ind_list:
            _img = hhi.gfi(img, ind)
            _label = 0
            self.notissue +=1
            data.append([_img, _label])
        return data

    def load_tissue(self, img_name):
        
        area = ImageDataset.SIZE**2
        data = []
        img = hi.cv2_reader(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        patch_ind_list = hhi.gen_patches_index(img.shape, 
                                                img_size=ImageDataset.SIZE, 
                                                stride=ImageDataset.STRIDE, 
                                                keep_last_size=True)
        # print(len(patch_ind_list))
        for ind in patch_ind_list:
            _img = hhi.gfi(img, ind)
            _gray = hhi.gfi(gray, ind)
            _label = 1
            # print(np.sum(_gray<245))
            if np.sum(_gray<ImageDataset.TISSUE_CUTOFF) < area*0.5:
                _label = 0
            if _label>0:
                self.notissue +=1
            else:
                self.tissue +=1
            data.append([_img, _label])
        return data

    def load_pollution(self, img_name):
        area = ImageDataset.SIZE**2
        data = []
        img = hi.cv2_reader(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        folder, file_name = os.path.split(img_name)
        id_, _ = os.path.splitext(file_name)
        roi_point = rwr.get_qupath_roi(os.path.join(folder, id_+'.geojson'))
        roi = np.zeros(gray.shape, dtype=np.uint8)
        rwr.get_mask_from_qupath_point(roi, roi_point)
        patch_ind_list = hhi.gen_patches_index(img.shape, 
                                                img_size=ImageDataset.SIZE, 
                                                stride=ImageDataset.STRIDE, 
                                                keep_last_size=True)
        # print(len(patch_ind_list))
        for ind in patch_ind_list:
            _img = hhi.gfi(img, ind)
            _gray = hhi.gfi(gray, ind)
            _roi = hhi.gfi(roi, ind)
            _label = 1
            if  np.sum(_roi>0)>area*0.4:
                _label = 0
            elif np.sum(_gray<ImageDataset.TISSUE_CUTOFF) < area*0.5:
                _label = 0
            if _label>0:
                self.notissue +=1
            else:
                self.tissue +=1
            data.append([_img, _label])
        return data


    def __len__(self):
        return len(self.data_trian)

    def __getitem__(self, idx):
        patch, label = self.data_trian[idx]
        img = Image.fromarray(patch)

        if self.transforms is not None:
           return self.transforms(img), torch.LongTensor([label])

        return img, torch.LongTensor([label])

def just_and_create(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_model(n_class):
    if MODEL == 'resnet50':
        model       = tm.resnet50(pretrained=True)
        model.fc    = nn.Linear(2048 , n_class)
    elif MODEL == 'resnet34':
        model       = tm.resnet34(pretrained=True)
        model.fc    = nn.Linear(512 , n_class)
    elif MODEL == 'resnet18':
        model       = tm.resnet18(pretrained=True)
        model.fc    = nn.Linear(512 , n_class)
    elif MODEL == 'vgg16':
        model       = tm.resnet34(pretrained=True)
        model.classifier = nn.Sequential(nn.Linear(25088 , 4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5,inplace=False),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5,inplace=False),
                                nn.Linear(4096, n_class)
                            )
    return model

def train_epoch(V,model, loss_fn, optimizer,dataloader_train,args):
    model.train()

    global train_num,train_loss,num
    pred = []
    label = []
    for step in tqdm(dataloader_train):
        tissue = []
        notissue = []
        data_train=[]
        target_train=[]
        
        data_train, target_train = step#next(dataiter_train)
        # target_train=[np.argmax(hot) for hot in target_train]
        # target_train=torch.Tensor(target_train).long()
        # target_train = target_train.cuda()
        data_train = Variable(data_train.float().cuda())
        target_train = Variable(target_train.cuda())
        output = model(data_train)
        
        output = torch.squeeze(output) 
        # probs = output.sigmoid()
        loss = loss_fn(output, target_train.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for i in range(len(output)):
            pred.append(np.argmax(output[i].squeeze().cpu().detach().numpy()))
        label.extend(target_train.squeeze().cpu().detach().numpy())

        train_num.append(num)
        train_loss.append(loss.item())
        num +=1

        
        if random.randint(0,50) < 5:
            for i in range(len(label)):
                if len(notissue)>=16 and len(tissue)>=16:
                    break
                if target_train.squeeze()[i]==0 and len(notissue)<16:
                    notissue.append(data_train[i].squeeze().cpu().detach().numpy())
                if target_train.squeeze()[i]==1 and len(tissue)<16:
                    tissue.append(data_train[i].squeeze().cpu().detach().numpy())
            tissue = hi.concat_img(tissue, nrow=4, channel_frist=True)
            notissue = hi.concat_img(notissue, nrow=4, channel_frist=True)
            V.vis_line(train_num,train_loss,win = 0,opts=\
                dict(title='LOSS of training, {} num each epoch'.format(len(dataloader_train)//20)))
            V.vis_img(torch.FloatTensor(tissue), win=1, opts=dict(title='tissue'))
            V.vis_img(torch.FloatTensor(notissue), win=2, opts=dict(title='notissue'))
    acc = accuracy_score(label,pred)
    rec = recall_score(label,pred,average='macro')
    pre = precision_score(label,pred,average='macro')
    info("ACC:{:.3} REC:{:.3} PRE:{:.3}".format(acc,rec,pre))
    #return model

def valid_epoch(V,model, loss_fn, dataloader_valid, args, epoch):
    print('Begining validation of epoch {}'.format(epoch))
    global best_f1
    model.eval()
    steps = len(dataloader_valid)
    dataiter_valid = iter(dataloader_valid)

    tmp_output = np.zeros((args.n_item_val,args.n_class))
    tmp_label = np.zeros((args.n_item_val,args.n_class))

    point = 0

    for step in tqdm(dataiter_valid):
        data_valid, target_valid = step#next(dataiter_valid)
        target_valid_copy=target_valid
        target_valid=[np.argmax(hot) for hot in target_valid]
        target_valid=torch.Tensor(target_valid).long()
        with torch.no_grad():
            data_valid = Variable(data_valid.float().cuda())

        target_valid = Variable(target_valid.cuda())

        batch_size = len(data_valid)
        output = model(data_valid)
        output = torch.squeeze(output)
        probs = output.sigmoid()
        #loss = loss_fn(probs, target_valid)
        target_valid=target_valid_copy
        probs = probs.cpu().detach().numpy()
        target_valid =  target_valid.cpu().detach()
    
        tmp_output[point:(point+batch_size),:] = probs
        tmp_label[point:(point+batch_size),:] = target_valid
        #break

    pred = np.zeros(tmp_output.shape[0])
    lab = np.zeros(tmp_label.shape[0])
    for i in range(len(tmp_label)):
        pred[i] = np.argmax(tmp_output[i])
        lab[i] = np.argmax(tmp_label[i])
    acc = accuracy_score(lab,pred)
    rec = recall_score(lab,pred,average='macro')
    pre = precision_score(lab,pred,average='macro')
    f1 = f1_score(lab,pred,average='macro')
    print('F1 score is :', f1)
    #torch.save(model, './checkpoints/res/res_epoch_'+str(epoch)+'.pth')
    if f1 > best_f1:
        best_f1 = f1
        just_and_create(os.path.join(args.weight_path,args.stain_))
        torch.save(model,os.path.join(args.weight_path,args.stain_,'Epoch_{}_F1_{:.04}_acc_{:.04}.pth'.format(epoch,f1,acc)))
        print('ACC :',acc)
        print('F1:',f1)
        print('Best f1 score is updated.')
    #return best_f1
    
def pretraining(args):
    transform = transforms.Compose([
                                    transforms.RandomResizedCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04),
                                    transforms.ToTensor()
                                    ])
    V = vis.visdom_vis(env_name='glioma_tissue-seg-'+MODEL)
    train_data = ImageDataset(transforms=transform)
    # valid_data = ImageDataset(data_path = '/media/chuhan/M008/zhaoke_data/zhaoke',img_size=224,n_class=args.n_class,type_ = 'valid',normalize=True)

    print('Len of train_data: {}'.format(len(train_data)))

    args.n_item_train   = len(train_data)
    # args.n_item_val     = len(valid_data)
    loss_fn             = CrossEntropyLoss().cuda()
    trainloader         = DataLoader(train_data,batch_size=256,\
                                    shuffle=True, drop_last=True, num_workers=16)
    model               = load_model(args.n_class).cuda()
    # optimizer           = SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0.0001)
    optimizer           = torch.optim.Adam(model.parameters(), lr=0.0001)
    # valloader           = DataLoader(valid_data,batch_size=32,shuffle=True,num_workers=7)

    for epoch in range(args.epoch):
        print('Epoch is:',epoch)
        train_epoch(V,model,loss_fn, optimizer,trainloader,args)
        if epoch>=10:
            torch.save(model.state_dict(), os.path.join(args.weight_path,'epoch-{}.pth'.format(epoch)))
        # valid_epoch(V,model, loss_fn, valloader, args, epoch)
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    args.stain_ = 'nor'
    args.n_class = 2
    args.n_feature = 960
    args.model_name = 'res'
    args.weight_path = os.path.join('./weight', MODEL)
    just_ff(args.weight_path, create_floder=True)
    seed_enviroment(19940405)
    pretraining(args)
