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
from torchvision import transforms as tfs
#import kaolin.rep.TriangleMesh as Mesh

__all__ = ['ManoData', 'NormTransform', 'HandInfoTransform', 'ToTensorTransform']

uvz_g = np.zeros((256,256,3),dtype=np.float32)+1e-3
rgb_g = Image.fromarray(uvz_g.astype(np.uint8))



def grid_mask(img):
    if np.random.randint(0, 10)>30:
        w = img.shape[0]
        ww = math.ceil(1.5*w)
        d = np.random.randint(int(math.ceil(96*w/224)), w)
        l = min(max(int(d*0.5+0.5),1),d-1)
        mask = np.ones((ww,ww), np.uint8)
        st_w = np.random.randint(d)
        st_h = np.random.randint(d)
        for i in range(-1, ww//d+1):
            s = d*i + st_w
            t = s + l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[s:t,:] = 0
            s = d*i + st_h
            t = s + l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[:,s:t] = 0
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.rotate(np.random.randint(360))
        mask = np.asarray(mask)
        mask = mask[(ww-w)//2:(ww-w)//2+w, (ww-w)//2:(ww-w)//2+w]
        mask = 1-mask.reshape(w,w,1)
        img = img * mask 
    return img


def random_crop(rgb, mask, uvz):
    if np.random.randint(0, 3)>30:
        w=rgb.shape[0]
        x = np.random.randint(5, w//4)
        #y = np.random.randint(1, w//8)
        rgb  = cv2.resize(rgb[x:-x,x:-x],(w,w))
        mask = cv2.resize(mask[x:-x,x:-x],(w,w))
        scale = 1 + 2*x/w
        uvz[...,:-1]  = scale*(uvz[...,:-1] - 0.5)+0.5
        uvz[...,-1] = scale * uvz[...,-1]
        
        #rgb  = cv2.resize(rgb[y:y+d,x:x+d],(w,w))
        #mask = cv2.resize(mask[y:y+d,x:x+d],(w,w))
        if mask.shape[-1]>3:
            mask = mask[...,None]
    return rgb, mask,uvz


def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(256),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        #tfs.RandomGrayscale(p=0.1),
        tfs.ToTensor()
        #tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    im_aug = tfs.Compose([
        tfs.ToTensor()
        #tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x


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



def rgb_uvz_mask_loader(path):
    try:
        info = np.load(path)
        rgb = info['rgb']
        if 'rgb_hand' in info.keys():
            rgb = rgb if random.uniform(0, 1)>0.5 else info['rgb_hand']
        mask = Image.open(io.BytesIO(info['mask'])) if 'mask' in info.keys() else None
        rgb, uvz = Image.open(io.BytesIO(rgb)),info['uvz']
    except:
        rgb,uvz,mask = rgb_g,uvz_g, rgb_g
        print(path)
        pass
    return rgb, uvz, mask

def rgb_uvz_loader(path):
    try: 
        info = np.load(path)
        rgb = info['rgb']
        if 'rgb_hand' in info.keys():
            rgb = rgb if random.uniform(0, 1)>0.5 else info['rgb_hand']
        rgb, uvz = Image.open(io.BytesIO(rgb)),info['uvz']
    except:
        rgb,uvz = rgb_g,uvz_g
        print(path)
        pass
    return rgb, uvz

def rgb_mask_loader(path):
    info      = np.load(path)
    rgb, mask = info['rgb'],info['mask']
    rgb, mask = Image.open(io.BytesIO(rgb)), Image.open(io.BytesIO(mask))
    return rgb, mask

def read_file_list(data_dir,fn):
    
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'    
    file_list=[]
    if not os.path.exists(fn):
        print(fn,'is not exists')
        return file_list
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            rgb_fn = data_dir + line.strip()
            #num = 0
            #if 'xvx_frei_130k' in rgb_fn:
            #    num = int(os.path.basename(rgb_fn)[2:-4]))*0
            #if num<33000:
            file_list.append(rgb_fn)
    random.shuffle(file_list)
    return file_list


def get_project_map(uvz, camera_project, size, crop_r=1.2, corr_mask=None):
    
    corr_mask=corr_mask.astype(bool) 
    vertices = uvz[corr_mask,:]
    word_xyz = np.concatenate((vertices, np.ones_like(vertices[:,-1:])),axis=-1)
    uvs = camera_project.dot(word_xyz.transpose(1,0)).transpose(1,0)
    uv = uvs[:,:-1]/(uvs[:,-1:]+1e-8)
    max_uv, min_uv = np.max(uv,axis=0), np.min(uv,axis=0)
    center = (max_uv + min_uv)/2
    w   = np.max(max_uv - min_uv, axis=-1)
    top_left = center - w * abs(crop_r)/2
    if crop_r>1: 
        uv = uv - top_left
        top_left = np.min(uv,axis=0)


    uv = np.clip(np.array(uv + 0.5).astype(np.int32),0, size[0]-1)
    mask = np.zeros(size,dtype=np.uint8)
    index = (uv[...,1],uv[...,0])
    mask[index] = 1

    if crop_r > 0:
        x,y,z=[uvz[:,:,i] for i in range(3)]
        uvz = np.stack([x,-y,-z],axis=-1)
        vertices = uvz[corr_mask,:]
        min_max = [np.min(vertices,axis=0),np.max(vertices,axis=0)]
        w_v = max((min_max[1]-min_max[0])[:-1]) 
        s = w/(w_v+1e-8)
        uvz = (uvz - min_max[0]) * s
        top_left = np.min(uv,axis=0)
        uvz[...,:-1] += top_left
        uvz /= size[0]
        
        #top_left = np.min(uv,axis=0)
        #min_max = [np.min(vertices,axis=0),np.max(vertices,axis=0)]
        #uvz = (uvz - min_max[0])/(min_max[1]-min_max[0] + 1e-8)
        #uvz[...,:-1] = (top_left + uvz[...,:-1] * w) / size[0]
        #uvz[...,-1] = uvz[...,-1] * w/size[0]

    #vertices = uvz[corr_mask,:]
    #uv = vertices[:,:-1] * size[0]
    #uv = np.clip(np.array(uv + 0.5).astype(np.int32),0, size[0]-1)
    #mask = np.zeros(size,dtype=np.uint8)
    #index = (uv[...,1],uv[...,0])
    #mask[index] = 1
     
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8),iterations = 1)            
    #mask = mask-cmask
    return mask[...,None].astype(np.float32),uvz


def rotation_data(uvz,rgb, mask, corr_mask,stage=8):
    i = np.random.randint(0,360)
    w = rgb.shape[0]
    M = cv2.getRotationMatrix2D((w//2,w//2),i,1)
    rgb = cv2.warpAffine(rgb,M,(w,w))
    mask = cv2.warpAffine(mask,M,(w,w))
    if len(mask.shape)<3:
        mask=mask[...,None].astype(np.float32)
    if len(corr_mask.shape)<3:
        corr_mask = corr_mask[...,None].astype(np.float32)
    uvz[...,:-1] = np.matmul(np.concatenate((uvz[...,:-1]*w, np.ones_like(uvz[...,-1:])),axis=-1),M.T)/w

    if np.random.randint(0,10)>20:
        M = np.float32([[-1, 0, w], [0, 1, 0]])
        rgb = cv2.warpAffine(rgb,M,(w,w))
        mask = cv2.warpAffine(mask,M,(w,w))
        if len(mask.shape)<3:
            mask=mask[...,None].astype(np.float32)
        uvz[...,:-1] = np.matmul(np.concatenate((uvz[...,:-1]*w, np.ones_like(uvz[...,-1:])),axis=-1),M.T)/w

    return uvz * corr_mask, rgb, mask


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
        self.grid = np.stack((grid.T, grid),axis=-1)/size #size,size,2
        corr_mask = np.sum(np.asarray(Image.open('./HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1) > 0
        self.corr_mask = corr_mask.astype(np.float32)
        self.mode=mode 
        self.datasets = []
        for name in datanames.split(','):
            if name =='syth':
                data_dir = os.path.join(root, 'hand/')
            else:
                data_dir = os.path.join(root, 'hand/real/')
            if name == 'freihand_eval':
                self.datasets += glob.glob(data_dir+name+'/*.npz', recursive=True)
            else:
                data_list_fn = '{}/{}/{}_{}.txt'.format(data_dir,name,name,mode)
                self.datasets +=  read_file_list(data_dir, data_list_fn)
                if mode!='test':
                   data_list_fn = '{}/{}/{}_val.txt'.format(data_dir,name,name)
                   self.datasets +=  read_file_list(data_dir, data_list_fn)
        
        print(len(self.datasets))
        random.shuffle(self.datasets)
        #for fn in self.datasets:
        #    dar=rgb_uvz_loader(fn)
        #return 
                
        manager = Manager()
        self.datasets =  manager.list(self.datasets)
        self.size = size
        #bg_root=None
        if bg_root is not None:
            manager1 = Manager()
            self.backgrounds = glob.glob(f'{bg_root}/*/*.jpg', recursive=True)
            print('bg:',len(self.backgrounds))
            self.backgrounds =  manager1.list(self.backgrounds)
        else:
            self.backgrounds = None

        
        self._image_trans    = image_transform
        self._uvz_transform  = uvz_transform
    

    def camera_data_loader(self, sample):
        
        bf = 1 if 'syth' in sample else 0
        flag = 1.#0.05 if 'hand_HUMBCP' in sample or 'cmu' in sample else 0.1

        rgb, uvz, mask = rgb_uvz_mask_loader(sample)
        #mask = None
        #if mask is None:
        camera_project = np.array([[480, 0,-128,0],[0, -480,-128,0],[0, 0,-1,0]],dtype=np.float32)
        crop_r = 1 if rgb.size[0]==self.size else 1.2
        mask_, uvz = get_project_map(uvz, camera_project, rgb.size, crop_r, self.corr_mask)
        #mask = mask_
        if mask is None:
            mask = mask_
        else:
            mask = np.asarray(mask,dtype=np.uint8)
            if len(mask.shape)<3:
                mask = mask[...,None]
            mask = np.sum(mask,axis=-1,keepdims=True)>0
            mask = mask.astype(np.float32)
        
        grid_flag = 0 if np.sum(mask) < 100 else 1.
            
        '''resize'''
        if self.backgrounds:
            algha = np.random.randint(0,10)/10. if self.mode=='train' else 1.
            background = pil_loader(random.choice(self.backgrounds))
            front_w, front_h = self.size,self.size
            low_ratio = max(front_w/background.size[0], front_h/background.size[1])
            high_ratio = max(min(5, low_ratio*10), 1)
            ratio = random.uniform(low_ratio, high_ratio)
            background = background.resize((math.ceil(background.size[0]*ratio), math.ceil(background.size[1]*ratio)))
            crop_x, crop_y = random.randint(0, background.size[0]-front_w), random.randint(0, background.size[1]-front_h)
            bg = background.crop((crop_x, crop_y, crop_x+front_w, crop_y+front_h))
            bg  = np.asarray(bg.resize((self.size, self.size)), dtype=np.float32)
            w = self.size
            rgb = rgb.resize((w,w))
            mask = cv2.resize(mask, (w,w)).reshape((w,w,1))
            rgb = np.asarray(rgb,dtype=np.float32)
            bg = bg * (1-mask) + rgb * mask if bf>0 or algha < 0.3 and grid_flag>0 else rgb
            rgb = bg/255.
        else:
            rgb = rgb.resize((self.size,self.size))
            rgb = np.asarray(rgb,dtype=np.float32)/255.
            mask = cv2.resize(mask, (self.size,self.size)).reshape((self.size,self.size,1))

        if self.mode=='train':
            uvz, rgb, mask = rotation_data(uvz,rgb, mask, self.corr_mask)
            rgb = grid_mask(rgb) 
            if crop_r==1.:
                rgb,mask,uvz = random_crop(rgb, mask,uvz) 
        uvz = uvz * self.corr_mask[...,None]
        grid = self.grid * mask + (1-mask)*2.

        grid = self._image_trans(grid) 
        mask = self._image_trans(mask)

        grid_flag = torch.tensor(np.array([grid_flag],dtype=np.float32).reshape(1,1),dtype=torch.float32)

        flag = torch.tensor(np.array([flag],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'grid':grid,'mask':mask,'grid_flag':grid_flag,'flag':flag}

        return rgb, uvz, project_info


    def youtube_data_loader(self, sample):
        rgb, uvz = rgb_uvz_loader(sample)

        grid_flag,flag = [0.5,0.5] if 'youtube_linear_interpolation' in sample else [1,1]

        w, h = rgb.size
        dxy = np.random.randint(0, w//4, 2)
        left_top_x, right_bottom_x = w//4 - dxy[0], w//4 * 3 + dxy[1]
        crop_w = right_bottom_x - left_top_x    
        rgb = rgb.crop((left_top_x, left_top_x, left_top_x + crop_w, left_top_x + crop_w))

        uvz[:,:-1] -= left_top_x
        resize_ratio =  self.size/crop_w
        rgb = rgb.resize((self.size,self.size))
        uvz *= resize_ratio
        
        camera_project = np.array([[self.size,0,0,0],[0,self.size,0,0],[0,0,0,1]],dtype=np.float32)
        min_v = np.min(uvz,(0,1))
        uvz[...,-1] = uvz[...,-1] - min_v[-1]
        uvz = uvz * self.corr_mask[...,None]

        mask, uvz = get_project_map(uvz/self.size, camera_project, rgb.size, -1,self.corr_mask)
        
        grid_flag = 0 if np.sum(mask) < 100 else grid_flag

        rgb = np.asarray(rgb,dtype=np.float32)

        if self.backgrounds and grid_flag:
            algha = np.random.randint(0,10)/10. if self.mode=='train' else 1.
            background = pil_loader(random.choice(self.backgrounds))
            front_w, front_h = self.size,self.size
            low_ratio = max(front_w/background.size[0], front_h/background.size[1])
            high_ratio = max(min(5, low_ratio*10), 1)
            ratio = random.uniform(low_ratio, high_ratio)
            background = background.resize((math.ceil(background.size[0]*ratio), math.ceil(background.size[1]*ratio)))
            crop_x, crop_y = random.randint(0, background.size[0]-front_w), random.randint(0, background.size[1]-front_h)
            bg = background.crop((crop_x, crop_y, crop_x+front_w, crop_y+front_h))
            bg  = np.asarray(bg.resize((self.size, self.size)), dtype=np.float32)
            rgb = bg*(1-mask) + rgb * mask if algha < 0.3 and grid_flag else rgb

        rgb/=255.
        if self.mode=='train':
            uvz, rgb, mask = rotation_data(uvz,rgb, mask, self.corr_mask)
            rgb = grid_mask(rgb)
            #rgb,mask,uvz = random_crop(rgb, mask,uvz)
        uvz = uvz * self.corr_mask[...,None]
        grid = self.grid * mask + (1-mask)*2

        grid = self._image_trans(grid) 
        mask = self._image_trans(mask)

        grid_flag = torch.tensor(np.array([grid_flag],dtype=np.float32).reshape(1,1),dtype=torch.float32)

        flag = torch.tensor(np.array([flag],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'grid':grid,'mask':mask,'grid_flag':grid_flag,'flag':flag}

        return rgb, uvz, project_info

    def freihand_eval_loader(self, sample):
        rgb, mask = rgb_mask_loader(sample)
        rgb   = np.asarray(rgb.resize((self.size, self.size)), dtype=np.float32)/255.
        mask  = np.asarray(mask.resize((self.size,self.size)), dtype=np.float32)>0
        mask  = mask[...,None].astype(np.float32)
        grid = self.grid * mask + (1-mask)*2

        grid = self._image_trans(grid) 
        mask = self._image_trans(mask)

        grid_flag = torch.tensor(np.array([10],dtype=np.float32).reshape(1,1),dtype=torch.float32)

        flag = torch.tensor(np.array([0],dtype=np.float32).reshape(1,1),dtype=torch.float32)
        project_info = {'grid':grid,'mask':mask,'grid_flag':grid_flag,'flag':flag}

        return rgb, uvz_g, project_info


    def __getitem__(self, index):
        sample = self.datasets[index]
        if 'youtube' in sample:
            rgb, uvz, project_info = self.youtube_data_loader(sample)
        elif 'freihand_eval' in sample:
            rgb, uvz, project_info =  self.freihand_eval_loader(sample)
        else:
            rgb, uvz, project_info =  self.camera_data_loader(sample)            
        

        rgb=Image.fromarray((rgb*255).astype(np.uint8)) 
        rgb = train_tf(rgb) if self.mode=='train' else test_tf(rgb)
              
        #if self._image_trans is not None:
        #    rgb = self._image_trans(rgb)

        if self._uvz_transform is not None:
            uvz = self._uvz_transform(uvz)
        
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

