import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2
import utils
import glob

BATCH_SIZE=1
SIZE=256



def get_uv_project(vertices, project):
    word_xyz = np.concatenate((vertices, np.ones_like(vertices[...,-1:])),axis=-1)
    uvs = project.dot(word_xyz.transpose(1,0)).transpose(1,0)
    uv = uvs[...,:-1]/(uvs[...,-1:]+1e-8)
    return uv

def get_top_left(uv, crop_r=1.2):
    max_uv, min_uv = np.max(uv,axis=0), np.min(uv,axis=0)
    center = (max_uv + min_uv)/2
    size   = np.max(max_uv - min_uv)
    top_left = (center - size * crop_r/2 + 0.5).astype(np.int32)
    size = np.int32(size * crop_r + 0.5)
    return top_left, size
    

def process(image, vertices, project, extern=[1,1,1]):
    '''crop'''
    uv = get_uv_project(vertices, project)       
    top_left, size = get_top_left(uv)
    vertices = vertices * np.array(extern).reshape(1,3) * np.array([1,-1,-1]).reshape(1,3)
    image = image[top_left[1]:top_left[1]+size, top_left[0]:top_left[0]+size].copy()
    return image, vertices



v_to_vt, faces = utils._parse_obj('./HandRightUV.obj')


v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(1, axis=0)).cuda()

v_to_vt = 2 * (torch.cat((v_to_vt[...,1:2], v_to_vt[...,0:1]),-1)) - 1
camera_npz = np.load('./camera_info.npz')
camera_intern = torch.from_numpy(np.concatenate((camera_npz['camera_calib'], np.array([0.,0.,0.]).reshape(3,1)),axis=-1).astype(np.float32)[None,...].repeat(BATCH_SIZE, axis=0)).cuda()
camera_extern = torch.from_numpy(np.concatenate((camera_npz['camera_extrn'], np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0).astype(np.float32)[None,...].repeat(BATCH_SIZE, axis=0)).cuda()
project_mat = torch.matmul(camera_intern, camera_extern)
np_corr_mask = np.sum(np.asarray(Image.open('HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1,keepdims=True) > 0
corr_mask = np_corr_mask.astype(np.float32)[None,...].transpose(0,3,1,2)
corr_mask = torch.from_numpy(corr_mask.repeat(BATCH_SIZE, axis=0)).cuda().float()




rgb_fn = glob.glob(f'/data/hand/syth/rgb_256/*.jpg', recursive=True)
project = np.array([[480, 0,-128, 0],[0, -480, -128, 0], [0, 0, -1, 0]],dtype=np.float32).reshape(3,4)
print(len(rgb_fn))
cnt = 0
for fn in rgb_fn:
    uvz_fn = fn.replace('/rgb_256/','/uvz/')
    data = np.load(uvz_fn.replace('.jpg','.npz'))
    if 'arr_0' in data.keys():
        uvz = data['arr_0']
    elif 'uv_map' in data.keys():
        uvz = data['uv_map']
    else:
        continue
    uvz = torch.from_numpy(uvz.transpose(2,0,1).astype(np.float32)[None,...]).cuda()
    xyz = utils.sample_uv_xyz(v_to_vt, uvz)[0].cpu().detach().numpy()
    image = cv2.imread(fn)
    image, vertices = process(image, xyz,project,[1,-1,-1])
    h,w,c=image.shape
    if h<40 or w < 40 :
        continue
    
    save_fn  = fn.replace('/rgb_256/','/rgb/')
    cv2.imwrite(save_fn,image)
    if cnt % 1000==0: 
        print('{}/{}:{}'.format(cnt,len(rgb_fn),save_fn))
    cnt += 1
