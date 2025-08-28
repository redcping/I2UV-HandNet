# -*- coding:utf-8 -*-
from pathlib import Path
import pickle
from PIL import Image
from collections import namedtuple
import numpy as np
import os
import glob
import math
import random
import torch
import cv2
import torch.utils.data as data
from multiprocessing import Manager
#import kaolin.rep.TriangleMesh as Mesh

__all__ = ['ManoData', 'NormTransform', 'HandInfoTransform', 'ToTensorTransform']

def pil_loader(path, rgb=True):
    image = Image.open(path)
    if rgb:
        return image.convert('RGB')
    else:
        return image

def pickle_loader(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def uvz_loader(path):
    uvz=None
    info = np.load(path)
    if 'arr_0' in info.keys():
        uvz=info['arr_0']
    elif 'uv_map' in info.keys():
        uvz = info['uv_map']
    else:
        print(path)
        uvz = None
    return uvz

def data_loader(path):
    info = np.load(path)
    rgb, uvz = Image.open(io.BytesIO(info['rgb'])),info['uvz']
    return rgb, uvz

def read_file_list(data_dir,fn):
    
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'    
    file_list=[]
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            rgb_fn = data_dir + line.strip()
            uvz_fn = rgb_fn.replace('/rgb/','/uvz/').replace('.jpg','.npz')
            file_list.append([rgb_fn, uvz_fn])
    return file_list

class ManoData(data.Dataset):

    def __init__(self, root,
                 datanames,
                 mode ='train',
                 size = 256,
                 bg_root=None,
                 image_transform=None,
                 uvz_transform=None):
        super().__init__() 
        self.datasets = []
        for name in datanames.split(','):
            if name =='syth':
                data_dir = os.path.join(root, 'hand/')
            else:
                data_dir = os.path.join(root, 'hand/real/')
            data_list_fn = '{}/{}/{}_{}.txt'.format(data_dir,name,name,mode)
            self.datasets +=  read_file_list(data_dir, data_list_fn)
        
        print(len(self.datasets))
                
        manager = Manager()
        self.datasets =  manager.list(self.datasets)
        self.size = size


        if bg_root is not None:
            manager1 = Manager()
            self.backgrounds = glob.glob(f'{bg_root}/*/*.jpg', recursive=True)+ glob.glob(f'{bg_root}/*.jpg', recursive=True)
            print('bg:',len(self.backgrounds))
            self.backgrounds =  manager1.list(self.backgrounds)
        else:
            self.backgrounds = None

        
        self._image_trans    = image_transform
        self._uvz_transform  = uvz_transform
    

    def __getitem__(self, index):
        sample = self.datasets[index]
        rgb = pil_loader(sample[0])
        uvz = uvz_loader(sample[1])
    
        '''resize'''
        resize_ratio = self.size / rgb.size[0]
        
        if 'syth' in str(sample[0]) and self.backgrounds is not None:
            background = pil_loader(random.choice(self.backgrounds))
            front_w, front_h = rgb.size
            low_ratio = max(front_w/background.size[0], front_h/background.size[1])
            high_ratio = max(min(5, low_ratio*10), 1)
            ratio = random.uniform(low_ratio, high_ratio)
            background = background.resize((math.ceil(background.size[0]*ratio), math.ceil(background.size[1]*ratio)))
            crop_x, crop_y = random.randint(0, background.size[0]-front_w), random.randint(0, background.size[1]-front_h)
            mask = np.sum(np.asarray(rgb, dtype=np.uint8)>245, axis=-1,keepdims=True)
            mask = (mask<3).astype(np.uint8)
            kernel = np.ones((3,3),np.uint8) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            #mask = cv2.dilate(mask,kernel,iterations = 1)
            mask = mask[...,None].astype(np.uint8)
            rgb = np.asarray(rgb,dtype=np.uint8) * mask + (1.-mask)*np.asarray(background.crop((crop_x, crop_y, crop_x+front_w, crop_y+front_h)), dtype=np.uint8) 
            rgb = Image.fromarray(np.uint8(rgb))
         

        rgb = rgb.resize((self.size,self.size))
        rgb = np.asarray(rgb,dtype=np.float32)
        rgb = rgb/255.

       
        if self._image_trans is not None:
            rgb = self._image_trans(rgb)

        if self._uvz_transform is not None:
            uvz = self._uvz_transform(uvz)
        return rgb.float(), uvz.float(), torch.from_numpy(np.array([resize_ratio]).reshape(1,1)).float()

    def __len__(self):
        return len(self.datasets)



class NormTransform(object):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image):
        image = image - self.mean
        return image/(self.sigma+1e-6)


class HandInfoTransform(object):
    def __init__(self):
        super().__init__()
    def __call__(self, hand_info):
        verts_3d = torch.from_numpy(hand_info['verts_3d']).type(torch.float32)
        joints_2d, joints_3d = torch.from_numpy(hand_info['coords_2d']).type(torch.float32), torch.from_numpy(hand_info['coords_3d']).type(torch.float32)
        return {'joints_2d':joints_2d, 'joints_3d':joints_3d, 'verts_3d':verts_3d}


class ToTensorTransform(object):
    def __init__(self):
        super().__init__()
    def __call__(self, pic:np.ndarray):
        if pic.ndim == 2:
            pic = pic[:,:,None]
        return torch.from_numpy(pic.transpose(2,0,1))
