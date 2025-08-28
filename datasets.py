# -*- coding:utf-8 -*-
from pathlib import Path
import pickle
from PIL import Image
from collections import namedtuple
import numpy as np
import os,io
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

def rgb_uvz_loader(path):  
    info = np.load(path)
    rgb = info['rgb']
    if 'rgb_hand' in info.keys():
        rgb = rgb if random.uniform(0, 1)>0.5 else info['rgb_hand']
    rgb, uvz = Image.open(io.BytesIO(rgb)),info['uvz']
    return rgb, uvz

def read_file_list(data_dir,fn):
    
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'    
    file_list=[]
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            rgb_fn = data_dir + line.strip()
            file_list.append(rgb_fn)
    return file_list


def get_project_map(uvz, camera_project, size, crop_r=1.2):
    
    mask = np.sum(abs(uvz)>0,axis=-1)==3
    vertices = uvz[mask,:]
    word_xyz = np.concatenate((vertices, np.ones_like(vertices[:,-1:])),axis=-1)
    uvs = camera_project.dot(word_xyz.transpose(1,0)).transpose(1,0)
    uv = uvs[:,:-1]/(uvs[:,-1:]+1e-8)
    
    max_uv, min_uv = np.max(uv,axis=-1), np.min(uv,axis=-1)
    center = (max_uv + min_uv)/2
    size   = np.max(max_uv - min_uv, axis=-1)
    top_left = center - size * crop_r/2
    uv = uv - top_left
    uv = np.array(uv + 0.5).astype(np.int32)
    
    mask = np.zeros(size)
    index = [np.clip(uv[...,1],0,size[1]), np.clip(uv[...,0], 0, size[0])]
    mask[index] = 1
    return mask[...,None]

class ManoData(data.Dataset):

    def __init__(self, root,
                 datanames,
                 mode ='train',
                 size = 256,
                 bg_root=None,
                 image_transform=None,
                 uvz_transform=None):
        super().__init__()

        grid = np.array(range(size),dtype=np.float32).reshape(-1,1)
        grid = np.repeat(grid, size, axis=-1)
        self.grid = np.stack((grid, grid.T),axis=-1)/size #size,size,2

        self.mode=mode 
        self.datasets = []
        for name in datanames.split(','):
            if name =='syth':
                data_dir = os.path.join(root, 'hand/')
            else:
                data_dir = os.path.join(root, 'hand/real/')
            data_list_fn = '{}/{}/{}_{}.txt'.format(data_dir,name,name,mode)
            self.datasets +=  read_file_list(data_dir, data_list_fn)
        '''
        cnt = 0
        for sample in self.datasets:
            if '/rgb_uvz/' in sample:
                rgb, uvz = rgb_uvz_loader(sample)
            else:
                rgb_path = sample
                uvz_path = rgb_path.replace('/rgb/','/uvz/')[:-3]+'npz'
                rgb = pil_loader(rgb_path)
                uvz = uvz_loader(uvz_path)
            if len(rgb.size)<2 or len(uvz.shape)<3:
                cnt = cnt + 1
                print(sample)
        print(cnt)
        import sys
        sys.exit()
        '''
 
                
        manager = Manager()
        self.datasets =  manager.list(self.datasets)
        self.size = size

        if bg_root is not None:
            manager1 = Manager()
            self.backgrounds = glob.glob(f'{bg_root}/*/*.jpg', recursive=True)
            print('bg:',len(self.backgrounds))
            self.backgrounds =  manager1.list(self.backgrounds)
        else:
            self.backgrounds = None

        
        self._image_trans    = image_transform
        self._uvz_transform  = uvz_transform
    

    def camera_data_loader(self,sample):
        if '/rgb_uvz/' in sample:
            rgb, uvz = rgb_uvz_loader(sample)
        else:
            rgb_path = sample
            uvz_path = rgb_path.replace('/rgb/','/uvz/')[:-3]+'npz'
            rgb = pil_loader(rgb_path)
            uvz = uvz_loader(uvz_path)

        '''resize'''
        if self.backgrounds:
            algha = np.random.randint(8,10)/10. if self.mode=='train' else 1.
            background = pil_loader(random.choice(self.backgrounds))
            front_w, front_h = self.size,self.size
            low_ratio = max(front_w/background.size[0], front_h/background.size[1])
            high_ratio = max(min(5, low_ratio*10), 1)
            ratio = random.uniform(low_ratio, high_ratio)
            background = background.resize((math.ceil(background.size[0]*ratio), math.ceil(background.size[1]*ratio)))
            crop_x, crop_y = random.randint(0, background.size[0]-front_w), random.randint(0, background.size[1]-front_h)
            bg = background.crop((crop_x, crop_y, crop_x+front_w, crop_y+front_h))
            bg  = np.asarray(bg.resize((self.size, self.size)), dtype=np.float32)
            crop_r =1. if rgb.size[0]==self.size else 0.
            w   = random.randint(max(self.size//2,rgb.size[0]), max(self.size, rgb.size[0]))
            rgb = rgb.resize((w,w))
            resize_ratio = self.size / rgb.size[0]
            w, h = rgb.size
            x, y = random.randint(0, self.size-w), random.randint(0, self.size-h)
            rgb = np.asarray(rgb,dtype=np.float32)
            bg[y:y+h,x:x+w] = bg[y:y+h,x:x+w]*(1-algha) + algha * rgb
            rgb = bg/255.
        else:
            resize_ratio = self.size / rgb.size[0]
            crop_r = 1. if rgb.size[0]==self.size else 0.
            rgb = rgb.resize((self.size,self.size))
            resize_ratio = self.size / rgb.size[0]
            x,y=0,0
            rgb = np.asarray(rgb,dtype=np.float32)/255.

        resize_ratio = torch.tensor(np.array([resize_ratio],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        duv = torch.tensor(np.array([x,y],dtype=np.float32).reshape(1,2),dtype=torch.float32)/self.size
        crop_r = torch.tensor(np.array([crop_r],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        no_youtube = torch.tensor(np.array([1],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'resize_ratio':resize_ratio,'duv':duv, 'crop_r':crop_r,'no_youtube':no_youtube}
                
        
        return rgb, uvz, project_info


    def youtube_data_loader(self, sample):
        rgb, uvz = rgb_uvz_loader(sample)
        w, h = rgb.size
        dxy = np.random.randint(0, w//4, 2)
        left_top_x, right_bottom_x = w//4 - dxy[0], w//4 * 3 + dxy[1]
        crop_w = right_bottom_x - left_top_x    
        rgb = rgb.crop((left_top_x, left_top_x, left_top_x + crop_w, left_top_x + crop_w))
        uvz[:,:-1] -= left_top_x
        resize_ratio =  self.size/crop_w
        rgb = rgb.resize((self.size,self.size))
        uvz *= resize_ratio
        rgb = np.asarray(rgb,dtype=np.float32)/255.
        x,y,z=[uvz[:,:,i] for i in range(3)]
        uvz = np.stack([x,-y,-z],axis=-1) 
        resize_ratio = torch.tensor(np.array([resize_ratio],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        duv = torch.tensor(np.array([0,0],dtype=np.float32).reshape(1,2),dtype=torch.float32)/self.size
        crop_r = torch.tensor(np.array([0],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        no_youtube = torch.tensor(np.array([0],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'resize_ratio':resize_ratio,'duv':duv, 'crop_r':crop_r,'no_youtube':no_youtube}
        return rgb, uvz/self.size, project_info


    def __getitem__(self, index):
        sample = self.datasets[index]
        if 'youtube' in sample:
            rgb, uvz, project_info = self.youtube_data_loader(sample)
        else:
            rgb, uvz, project_info =  self.camera_data_loader(sample)            
              
        if self._image_trans is not None:
            rgb = self._image_trans(rgb)

        if self._uvz_transform is not None:
            uvz = self._uvz_transform(uvz)
        
        return rgb, uvz, project_info



    def __getitem1__(self, index):
        sample = self.datasets[index]
        if '/rgb_uvz/' in sample:
            rgb, uvz = rgb_uvz_loader(sample)
        else:
            rgb_path = sample
            uvz_path = rgb_path.replace('/rgb/','/uvz/')[:-3]+'npz'
            rgb = pil_loader(rgb_path)
            uvz = uvz_loader(uvz_path)
    
        '''resize'''
        
        
        algha = np.random.randint(8,10)/10. if self.mode=='train' else 1.
        background = pil_loader(random.choice(self.backgrounds))
        front_w, front_h = self.size,self.size
        low_ratio = max(front_w/background.size[0], front_h/background.size[1])
        high_ratio = max(min(5, low_ratio*10), 1)
        ratio = random.uniform(low_ratio, high_ratio)
        background = background.resize((math.ceil(background.size[0]*ratio), math.ceil(background.size[1]*ratio)))
        crop_x, crop_y = random.randint(0, background.size[0]-front_w), random.randint(0, background.size[1]-front_h)
        bg = background.crop((crop_x, crop_y, crop_x+front_w, crop_y+front_h))
        bg  = np.asarray(bg.resize((self.size, self.size)), dtype=np.float32)
        crop_r =1. if rgb.size[0]==self.size else 0.
        w   = random.randint(max(self.size//2,rgb.size[0]), max(self.size, rgb.size[0]))
        rgb = rgb.resize((w,w))
        resize_ratio = self.size / rgb.size[0]
        w, h = rgb.size
        x, y = random.randint(0, self.size-w), random.randint(0, self.size-h)
        rgb = np.asarray(rgb,dtype=np.float32)
        bg[y:y+h,x:x+w] = bg[y:y+h,x:x+w]*(1-algha) + algha * rgb
        rgb = bg/255.

       
        if self._image_trans is not None:
            rgb = self._image_trans(rgb)

        if self._uvz_transform is not None:
            uvz = self._uvz_transform(uvz)
        
        resize_ratio = torch.tensor(np.array([resize_ratio],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        duv = torch.tensor(np.array([x,y],dtype=np.float32).reshape(1,2),dtype=torch.float32)/self.size
        crop_r = torch.tensor(np.array([crop_r],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'resize_ratio':resize_ratio,'duv':duv, 'crop_r':crop_r}
        return rgb, uvz, project_info






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
        return torch.tensor(pic.transpose(2,0,1),dtype=torch.float32)
