import numpy as np
import numpy.random as rand 
from scipy.stats import bernoulli 
import pdb
import glob
import os.path as osp
import pandas as pd
import skimage.io as sko 
from sklearn.model_selection import train_test_split

import torch 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

import PIL
from PIL import Image 
import pickle 

def colorDigit(img, color=[255, 0, 0]):
    
    img = img.numpy() 
    idx = img < 50
    assert(len(img.shape) == 2)
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img[idx] = color
    
    return img

# if __name__ == '__main__':
#     # x = np.zeros( (32, 32), dtype=np.uint8)
#     # x[:4,:4] = 255
#     x = sko.imread('sample_image.png')
#     x = colorDigit(x[:, :, 0])
#     sko.imsave('test.png', x)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        try:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        except:
            print('Could not add Gaussian Noise')
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def trans_noisy_MNIST(x,rot,tra):

    tsfrm = transforms.ToPILImage()
    img_org = tsfrm(x)

    # img = img_org.transform(img_org.size, Image.AFFINE, (1,0,0,0,1,tra*128))
    img = img_org.transform(img_org.size, Image.AFFINE, (1,0,tra*128,0,1,0))
    img = img.rotate(angle=rot)

    tsfrm = transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                AddGaussianNoise(0,0.2)])
    return tsfrm(img)

def trans_MNIST(x, rot=0, tra=0):
    # tsfrm = transforms.Compose([transforms.ToPILImage(),
    #                             transforms.Resize((128, 128))])
    tsfrm = transforms.ToPILImage()
    img_org = tsfrm(x)

    # img = img_org.transform(img_org.size, Image.AFFINE, (1,0,0,0,1,tra*128))
    img = img_org.transform(img_org.size, Image.AFFINE, (1,0,tra*128,0,1,0))
    img = img.rotate(angle=rot)
    to_ten = transforms.ToTensor()
    
    return(to_ten(img))

def trans_col_MNIST(x,color):
    x = colorDigit(x,color)
    tsfrm = transforms.ToPILImage()
    img = tsfrm(x)
    to_ten = transforms.ToTensor()
    
    return(to_ten(img))

class MNIST(Dataset):

    def __init__(self, labels, transform, conf = 0.5, conf_type = 'rot', 
                img_pth ='/scratch/gobi2/sindhu/datasets/MNIST/processed', data_ty ='training', per_digit=False): 
        self.pth = img_pth  
        self.labels = labels 
        self.per_digit= per_digit
        self.tras = transform
        self.input_shape = 128 
        self.conf = conf
        self.conf_type = conf_type
        self.pallet = [[255,0,0],[0,255,0],[0,0,255],[221,222,0],[255,0,255],[0,255,255],[255,108,0],[255,0,118],[135,0,255],[117,177,177]]
        # self.pallet = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        self.data, self.targets = torch.load(osp.join(self.pth,f'{data_ty}.pt'))

    def __len__(self):
        return len(self.labels)

    def per_digit_idx(self):

        per_digit_indices = [] 
        for i in range(10):
            per_digit_indices.append(np.where(self.labels[:,1] == i)[0][10])      
        
        return per_digit_indices
    
    def load_image(self, idx):
        
        lbl_ind = int(self.labels[idx][1])
        img = self.data[self.labels[idx][0]]
        im_name = f'{self.labels[idx][0]}_{self.labels[idx][1]}_{self.labels[idx][2]}'
        u = int(self.labels[idx][2])
            
        if self.conf_type == 'bri': 
            im = self.tras(img, rot=0, tra=0)
            im = im.repeat(3,1,1)
            v = torch.ones([3,self.input_shape,self.input_shape])
            b = (self.conf)*torch.ones([3,self.input_shape,self.input_shape])     
            if u == 1: 
                im = torch.min((im+b),v)  
        
        if self.conf_type == 'rot':
            if u == 1:
                im = self.tras(img,self.conf,tra=0)
            elif u == 0:
                im = self.tras(img,rot=0,tra=0)
                
        if self.conf_type == 'trans': 
            if u == 1:
                im = self.tras(img,0,tra=self.conf)
            elif u == 0:
                im = self.tras(img,0,tra=0)

        if self.conf_type == 'heal': 
            if u == 1:
                im = self.tras(img,self.conf,0)
                im[0,20:25,20:25] = 0.95
            elif u == 0:
                im = self.tras(img,0,0)

        if self.conf_type == 'noisy':  
            if u == 1:
                im = self.tras(img,self.conf,0)
            elif u == 0:
                im = trans_MNIST(img,0,0)
        
        if self.conf_type == 'colour': 
            
            if u == 1: 
                im = self.tras(img, self.pallet[lbl_ind])
            if u == 0: 
                im = self.tras(img, self.pallet[rand.randint(0,9)])
            # im_path = osp.join(f'{im_name}_{u}.jpg')
            # sko.imsave(im_path, im.permute(1,2,0).cpu().numpy())
            # import pdb; pdb.set_trace()
                   
        return im, im_name

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
      
        if self.per_digit:
            per_didgit_indices = self.per_digit_idx()
            im, im_name = self.load_image(per_didgit_indices[idx])
            lbl = torch.tensor(int(self.labels[per_didgit_indices[idx]][1]))

        else: 
            im, im_name = self.load_image(idx)
            lbl = torch.tensor(int(self.labels[idx][1]))
        
        return im, lbl


def data_lab_MNIST(split='train'):
    
    root = '/scratch/gobi2/sindhu/datasets/MNIST/processed'
    
    if split == 'train':
        data_file = 'training.pt'
        
        _, targets = torch.load(osp.join(root, data_file))
        
        # for all classes 
        y = targets.numpy()  
        ind = np.arange(len(y))
        
        # for classes 2 and 6 
        # i = np.where(targets == 2)[0]; j = np.where(targets == 6)[0]
        # y_0 = np.zeros(len(i), dtype=int); y_1 = np.ones(len(j), dtype=int)
        
        # ind = list(i) + list(j)
        # y = list(y_0) + list(y_1)

        ind_tr, ind_val, y_tr, y_val= train_test_split(ind, y, test_size=0.05, random_state=42)

        return dict(zip(ind_tr,y_tr)), dict(zip(ind_val,y_val)) 
    
    elif split == 'test':
        data_file = 'test.pt'
        _, targets = torch.load(osp.join(root, data_file))

        # for all classes  
        y = targets.numpy() 
        ind = np.arange(len(y))

        # for classes 2 and 6 
        # i = np.where(targets == 2)[0]; j = np.where(targets == 6)[0]
        # y_0 = np.zeros(len(i), dtype=int); y_1 = np.ones(len(j), dtype=int)
        
        # ind = list(i) + list(j)
        # y = list(y_0) + list(y_1)

        return dict(zip(ind, y)) 

def add_conf(index_n_labels,p,qyu,N):

    # pdb.set_trace()
    # pu_y = np.array([1-qyu, qyu])

    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))

    py_u = np.array([np.repeat(1-qyu, len(np.unique(lab_all))), np.repeat(1-qyu, len(np.unique(lab_all)))])

    # pdb.set_trace()
                
    N_bar = len(lab_all)
    
    Ns = int(0.1*N) 

    Y = np.zeros(N)
    U = rand.binomial(1,0.96,N)

    for i in range(N):
        Y[i] = np.where(rand.multinomial(1,py_u[U[i]]))[0][0]
    
    idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0]) + list(np.where(Y==2)[0]) + list(np.where(Y==3)[0]) + list(np.where(Y==4)[0]) + \
        list(np.where(Y==5)[0]) + list(np.where(Y==6)[0]) + list(np.where(Y==7)[0]) + list(np.where(Y==8)[0]) + list(np.where(Y==9)[0]) 
    
    idx = list(np.where(lab_all==0)[0][:Ns]) + list(np.where(lab_all==1)[0][:Ns]) + list(np.where(lab_all==2)[0][:Ns]) + list(np.where(lab_all==3)[0][:Ns]) + \
            list(np.where(lab_all==4)[0][:Ns]) + list(np.where(lab_all==5)[0][:Ns]) + list(np.where(lab_all==6)[0][:Ns]) + list(np.where(lab_all==7)[0][:Ns]) + \
                list(np.where(lab_all==8)[0][:Ns]) + list(np.where(lab_all==9)[0][:Ns]) 

    Y = Y[idn];  U = U[idn]  
    
    # pdb.set_trace()    
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) 
    idx = np.array(idx)    

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf]).transpose(1,0)

    return labels_conf   


# def add_conf(index_n_labels,p=0.5,qyu=0.95,N=8000):
    
#     if qyu<0:
#         pu_y = np.array([-qyu, 1+qyu])
#     else:    
#         pu_y = np.array([1-qyu, qyu])
    
#     lab_all = np.array(list(index_n_labels.values()))
#     filename = np.array(list(index_n_labels.keys()))        
    
#     N_bar = len(lab_all)

#     # sampling U (guassian with variance 5)
#     Y = rand.binomial(1,p,N)
#     U = rand.binomial(1,pu_y[Y])
    
#     p = sum(Y)/N 
#     Ns1 = int(p*N)
#     Ns0 = N - Ns1 

#     idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0])
#     idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

#     Y = Y[idn]; U = U[idn]
    
#     U = np.array(U, dtype=int)
#     Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
#     yr = np.unique(Y)
#     ur = np.unique(U)

#     idx = np.array(idx)    
    
#     # confounded data 
#     filename_conf = filename[idx]
#     Y_conf = Y; U_conf = U
#     labels_conf = np.array([filename_conf, Y_conf, U_conf]).transpose(1,0)
    
#     return labels_conf



# if __name__ == "__main__":

#     index_n_labels, index_n_labels_v =  data_lab_MNIST(split='train')
#     labels_conf, labels_deconf = cb_backdoor(index_n_labels,p=0.5,qyu=0.95,N=9300)
#     train_data = MNIST(labels = labels_conf, cb_type= 'back', conf=0, conf_type= 'heal',
#                                 transform = trans_noisy_MNIST, data_ty='training')
#     train_loader = DataLoader(train_data, batch_size=1, shuffle=True) # set shuffle to True
#     print(train_data[0])

    # pdb.set_trace() 

    # im_path = osp.join('utils/vision_data_0.jpg')
    # sko.imsave(im_path, tr_data[0][0].permute(1,2,0).cpu().numpy())

    # pdb.set_trace()



