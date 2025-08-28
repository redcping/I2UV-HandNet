# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from models import Model
#from models_resnet import Resnet18_md
import numpy as np
import utils
import argparse
import cv2
import open3d as o3d
import torch.nn as nn
import json
import metrics
from torchvision import transforms as tfs

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Directory for storing input data')
parser.add_argument('--output_dir', type=str, help='Directory for storing output data')
parser.add_argument('--model_path', type=str,  help='path for storing model', default='D:/google下载/hand_recon_parameters_epoch28_new_z_best.pth')
parser.add_argument('--gpu_list', type=str,  help='gpu_list', default='0')
parser.add_argument('--batch_size', type=int,  help='gpu_list', default=1)
parser.add_argument('--do_uvz',     type=bool, help="do uvz",   default=True)
parser.add_argument('--do_score',   type=bool, help="do score", default=False)
parser.add_argument('--do_min_max_v',   type=bool, help="do score", default=False)
parser.add_argument('--do_z',   type=bool, help="do score", default=True)
parser.add_argument('--encoder_name', type=str, help="encode net", default='resnet50')

parser.add_argument('--do_super',   type=bool, help="do super", default=False)
parser.add_argument('--do_gan',   type=bool, help="do gan", default=False)
parser.add_argument('--do_mano',   type=bool, help="do mano", default=False)
args = parser.parse_args()
device_ids=[int(v) for v in args.gpu_list]

'''just for render'''
window_size=256
CAM_FX = 480
CAM_FY = 480
''''''
#
def visualize_non_blocking(vis, pcds):
    for pcd in pcds:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def cb(vis, points):
    colors = points + np.array([0,255,0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    visualize_non_blocking(vis, [pcd])
    vis.remove_geometry(pcd)

def save_obj(outmesh_path, vertices, faces):
       # vertices -= vertices[268:269]
        ## Write to an .obj file
    outmesh_path = outmesh_path.replace('wa_rgb','obj').replace('rgb','obj')
    with open(outmesh_path, 'w') as fp:
        for v in vertices:
            fp.write( 'v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )


def read_obj(obj_file):
    with open(obj_file, 'r') as fin:
        lines = [l 
            for l in fin.readlines()
            if len(l.split()) > 0
            and not l.startswith('#')
        ]
    
    # Load all vertices (v) and texcoords (vt)
    vertices = []
    faces = []
    for line in lines:
        lsp = line.split()
        if lsp[0] == 'v':
            x = float(lsp[1])
            y = float(lsp[2])
            z = float(lsp[3])   
            vertices.append((x, y, z))
    # Stack these into an array
    vertices = np.vstack(vertices).astype(np.float32)
    return vertices


def drawmesh(mesh,v,viewer):
    
    # np.save('d:/src/hand-graph-cnntest/v2.npy',np.asarray(v))
    # v = np.matmul(view_mat, v.T).T
    # np.save('d:/src/hand-graph-cnntest/v0.npy',np.asarray(v))
    # print('t1 ',time.time()-t1)
    # v = mesh_smoother.process(v)

    # print(v.shape)
    #view_mat = np.asarray([[1,0,0],[0,-1,0],[0,0,-1]])
    #v = np.matmul(view_mat, v.T).T
    #v[:,0] = -v[:,0]
    
    #print(np.max(v,axis=0),np.min(v,axis=0))

    #v = v + np.array([0, 0, -3])
    # np.save('d:/src/hand-graph-cnntest/v3.npy',np.asarray(vt))
    # mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
##############################################open3d可视化
    viewer.update_geometry(mesh)
    viewer.poll_events()

def create_mesh_viewer():
    
    mesh = o3d.io.read_triangle_mesh("./handdemo.obj")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([228/255, 178/255, 148/255])
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
    width=window_size*4 + 1, height=window_size*4 + 1,
    window_name='VR-hand-First person',visible = True
    )
    viewer.add_geometry(mesh)
    # print(view_mat)

    view_control = viewer.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    #extrinsic[0:3, 3] = 0
    cam_params.extrinsic = extrinsic
    '''
    cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, CAM_FX, CAM_FY,
    window_size // 2, window_size // 2
    )
    '''
    cam_params.intrinsic.set_intrinsics(
    window_size+1, window_size+1, CAM_FX, -CAM_FY,
    -window_size // 2, -window_size // 2
    )
    #cam_params.intrinsic.set_intrinsics()
    # pdb.set_trace()
    view_control.convert_from_pinhole_camera_parameters(cam_params)
    view_control.set_constant_z_far(1000)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./render_option.json')
    viewer.update_renderer()
    return mesh, viewer

def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))
  

def image_resize(image, size=(256,256)):
    h, w, _ = image.shape
    s=4
    box = [w//2 - size[0]//s, h//2-size[1]//s, 2*size[0]//s, 2*size[1]//s]
    box = [max(v,0) for v in box]
    d   = min(min(box[2] + box[0], w - 1),min(box[3] + box[1], h - 1))
    x1, y1, x2, y2 = box[0], box[1], box[0] + d, box[1] + d
    crop = image[y1:y2, x1:x2].copy()
    crop = cv2.resize(crop, size)
    return crop, [x1, y1, x2, y2]
    

def viz_sample(plt, image, vertices, faces=None):
    """Visualize a sample from the dataset.
    Args:
        faces: MANO faces.
    """   
    
    plt.imshow(image)
    if faces is None:
        plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='green', markersize=1)
    else:
        plt.triplot(vertices[:, 0], vertices[:, 1], faces, lw=0.2)
    plt.show()
    plt.pause(0.1)    # pause 1 second
    plt.clf()


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.conv3(x)
        return x





class HandMeshTester(object):
    '''hand mesh training schedule'''
    def __init__(self,size=256):
        super().__init__()
        self.do_super = args.do_super
        #init model
        self.size=size
        v_to_vt,_ = utils._parse_obj('HandRightUV_3K.obj')
        _,faces = utils._parse_obj( 'HandRightUV.obj')
        v_to_vt = (v_to_vt * (self.size-1)+0.5).astype(np.int32)
        v_to_vt = v_to_vt.astype(np.float32)/(self.size-1)
        self.v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(args.batch_size, axis=0)).cuda()
        print(self.v_to_vt.size())
        self.v_to_vt = 2 * torch.cat((self.v_to_vt[...,1:2], self.v_to_vt[...,0:1]),-1) - 1
        self.manoJregressor = self.create_J21_regressor()
        self.faces = faces.astype(np.int32)
        #torch.from_numpy(faces.astype(np.float32)[None,...].repeat(args.batch_size, axis=0))
        self.model_name = os.path.basename(args.model_path.split(',')[0])
        camera_npz = np.load('camera_info.npz')
        self.camera_intern = torch.from_numpy(np.concatenate((camera_npz['camera_calib'], np.array([0.,0.,0.]).reshape(3,1)),axis=-1).astype(np.float32)[None,...].repeat(args.batch_size, axis=0)).cuda()
        self.camera_extern = torch.from_numpy(np.concatenate((camera_npz['camera_extrn'], np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0).astype(np.float32)[None,...].repeat(args.batch_size, axis=0)).cuda()
        corr_mask = np.sum(np.asarray(Image.open('HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1,keepdims=True) > 0
        corr_mask = corr_mask.astype(np.float32)[None,...].transpose(0,3,1,2)
        self.corr_mask = torch.from_numpy(corr_mask.repeat(args.batch_size, axis=0)).float().cuda()
        #self.corr_mask = torch.nn.functional.interpolate(self.corr_mask, scale_factor=self.size/256.*2, mode='nearest')
        self.l1_loss_fn =  metrics.l1_loss
        self.setup_models()
        self.load_model()
        
    def create_J21_regressor(self):
        #mano_to_21v= [0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3,744,333,443,554,672]  (>15从v中找，小于15从joint中找)
        J16_regressor = np.load('ManoJregressor.npy')
        #J16_regressor = J16_regressor.todense()
        zeros_regressor = np.zeros((5,778),dtype = J16_regressor.dtype)
        k = [745, 317, 444, 556, 673]
        for i in range(len(k)):
            zeros_regressor[i,k[i]] = 1.
        J21_regressor = np.vstack((J16_regressor,zeros_regressor))
        return J21_regressor
    
    
    def mask_uvz(self, uvz, mask):
        scale = (1.0 * uvz.size()[-1])/mask.size()[-1]
        mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
        return uvz * mask
    
    def begin_valid(self):
        self.model.eval()
    
    def setup_models(self):
        self.project_mat = torch.matmul(self.camera_intern, self.camera_extern)
        #self.project_mat=None
        if args.do_z:
            self.size=256
            self.project_mat = torch.from_numpy(np.array([[self.size,0,0,0],[0,self.size,0,0],[0,0,0,1]],dtype=np.float32)[None,...].repeat(args.batch_size, axis=0)).cuda()
        self.dict_info={}
        self.dict_info['encoder'] = {'num_in_layers':3,'encoder_name':args.encoder_name,'pretrained':False}
        self.dict_info['decoder'] = {'do_uvz':args.do_uvz,'do_score':args.do_score,'camera_intern_extern':self.project_mat,'do_z':args.do_z}
        print('args.do_mano',args.do_mano)
        if args.do_mano:
            self.dict_info['do_mano'] = True
            
        self.model = Model(self.dict_info).cuda()
        if self.do_super:
            self.model_super = SRCNN().cuda()


    def norm_uvz(self, uvz, mask, min_max_v=None):
        b, c, _, _ = uvz.size()
        scale = (1.0 * uvz.size()[-1])/mask.size()[-1]
        mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
        if min_max_v is None:
            min_v,_ = torch.min(uvz.view(b, c, -1),dim=-1)
            min_v = min_v.view(b,c,1,1)
            uvs = (1-mask) *  min_v + mask * uvz
            max_v,_ = torch.max(uvz.view(b, c, -1),dim=-1)
            max_v = max_v.view(b,c,1,1)
        else:
            min_v,max_v = min_max_v
        return (uvz-min_v)/(max_v-min_v + 1e-8)*mask

    def ex_model(self,k,ex=['corr_mask','v_to_vt','camera_intern_extern','project_mat']):
        for e in ex:
            if e in k:
                return False
        return True

    def load_model(self):

        model_path = args.model_path.split(',')[0]
        #'''
        #if self.dict_info['decoder']['do_score']:
        #    model_path = model_path[:-4]+'_score.pth'
        checkpoint = torch.load(model_path)
        pretrained_dict_ = checkpoint['state']
        #import pdb;pdb.set_trace()
        model_dict = self.model.state_dict()
        #import pdb;pdb.set_trace()
        
        ex = ['corr_mask','v_to_vt','camera_intern_extern','project_mat','mano_decoder','dis_layer']
        if args.do_mano:
            ex.remove('mano_decoder')
        if args.do_gan:
            ex.remove('dis_layer')
        
        pretrained_dict = {k.replace('module.',''):v for k, v in pretrained_dict_.items() if self.ex_model(k,ex)}
        print('pretrain nodes:',len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        if self.do_super:
            checkpoint = torch.load(args.model_path.split(',')[1])
            self.model_super.load_state_dict(checkpoint['state'])
    
    def get_norm_loss(self,v1,v2):
        min_max_v1 = [np.min(v1,axis=0),np.max(v1,axis=0)]
        min_max_v2 = [np.min(v2,axis=0),np.max(v2,axis=0)]
        v1 = (v1-min_max_v1[0])/(min_max_v1[1] - min_max_v1[0])
        v2 = (v2-min_max_v2[0])/(min_max_v2[1] - min_max_v2[0])
        v_loss = np.mean(np.abs(v1-v2))
        return v_loss
    
    
    def test_tf(self, x, size):
        im_aug = tfs.Compose([
            tfs.FiveCrop(size-size//8),
            #tfs.Lambda(lambda crops: torch.stack([tfs.Resize(256)(crop) for crop in crops])),
            tfs.Lambda(lambda crops: torch.stack([tfs.ToTensor()(crop) for crop in crops]))
        ])
        x = tfs.ToPILImage()(x)
        scale = size/(size-size//8)
        x_aug = torch.nn.functional.interpolate(im_aug(x), scale_factor=scale, mode='bilinear', align_corners=True)
        x = tfs.ToTensor()(x)
        x = torch.cat((x_aug,x[None,...]),dim=0)
        return x
    
    
    def eval(self, image):
        if args.batch_size==6:
            image = self.test_tf(image, image.shape[1])
        else:
            image = tfs.ToTensor()(image)
            image = image[None,...]
            #image = torch.from_numpy(image.astype(np.float32)[None,...].repeat(args.batch_size, axis=0))/255.
        run_info=dict(image=image.cuda(), project_mat=self.project_mat.cuda(), v_to_vt = self.v_to_vt.cuda(), corr_mask=self.corr_mask.cuda())
        tensor_preds = self.model.forward(run_info)
        uvz_pred = tensor_preds['uvz_preds'][0] * self.corr_mask
        
        #uvz_pred = torch.nn.functional.interpolate(uvz_pred, scale_factor=scale, mode='nearest')* self.corr_mask
        #size=512
        #scale = size/self.size
        #corr_mask = (torch.nn.functional.interpolate(self.corr_mask, scale_factor=scale, mode='bilinear',align_corners=True)==1.).float()
        
        # uvz_pred = torch.nn.functional.interpolate(uvz_pred, scale_factor=scale, mode='bilinear', align_corners=True)
        # corr_mask = torch.nn.functional.interpolate(self.corr_mask, scale_factor=scale, mode='nearest')
        # #uvz_pred = torch.nn.functional.interpolate(uvz_pred, scale_factor=scale, mode='nearest')
        # uvz_pred *= corr_mask
        
        # size=256
        # v_to_vt=self.v_to_vt_
        # v_to_vt = (v_to_vt * (size-1)+0.5).astype(np.int32)
        # v_to_vt = v_to_vt.astype(np.float32)/(size-1)
        # self.v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(args.batch_size, axis=0)).cuda()
        # self.v_to_vt = 2 * torch.cat((self.v_to_vt[...,1:2], self.v_to_vt[...,0:1]),-1) - 1
        
        #print(uvz_pred.shape)
        vertices_mano=None
        if args.do_mano:
            vertices_mano = tensor_preds['mano']['vertices'].cpu()
            vertices_mano = vertices_mano[0].detach().numpy()
            joints_mano = tensor_preds['mano']['joints'].cpu()
            joints_mano = joints_mano[0].detach().numpy()
            vertices_mano=dict(vertices=vertices_mano,joints=joints_mano)
        
        if args.do_min_max_v:
            min_max_v= [v.view(-1,max(v.shape),1,1) for v in tensor_preds[-1]]
            uvz_pred = (uvz_pred * min_max_v[0] + min_max_v[1]) * self.corr_mask
            #print([v[0].cpu().detach().numpy() for v in min_max_v])
           
        if self.do_super:
            vertices = utils.sample_uv_xyz(self.v_to_vt, uvz_pred)[:,:778]
            #print(vertices[:,:778].shape,torch.min(vertices[:,:778],dim=1)[0].shape)
            min_max_v = [torch.min(vertices,dim=1)[0].reshape(1,3,1,1),torch.max(vertices,dim=1)[0].reshape(1,3,1,1)]
            uvz_pred_norm = self.norm_uvz(uvz_pred, self.corr_mask, min_max_v)
            uvz_pred_norm = self.model_super.forward(uvz_pred_norm) * self.corr_mask
            uvz_pred = uvz_pred_norm * (min_max_v[1]-min_max_v[0]) + min_max_v[0]
            
        
        
        
        sampled_pred = torch.nn.functional.grid_sample(uvz_pred, self.v_to_vt)
        sampled_mean = torch.mean(sampled_pred, dim=-1, keepdim=True).repeat(1,1,1,sampled_pred.size()[-1])
        self.sampled_loss = self.l1_loss_fn(sampled_pred, sampled_mean).item()*1000
        
        
        vertices = utils.sample_uv_xyz(self.v_to_vt, uvz_pred)[:,:778]
        
        #min_max_v = [torch.min(vertices,dim=1)[0].view(args.batch_size,1,3),torch.max(vertices,dim=1)[0].view(args.batch_size,1,3)]
        #vertices = (vertices - min_max_v[0])/(min_max_v[1] - min_max_v[0] + 1e-8)
        #vertices = torch.mean(vertices,dim=0,keepdim=True)
        #print(self.sampled_loss)

        vertices = vertices.cpu()
        vertices = vertices[0].detach().numpy()
        if args.do_mano:
            self.v_loss = self.get_norm_loss(vertices,vertices_mano['vertices'])
        
        #v1=vertices
        #min_max_v1 = [np.min(v1,axis=0),np.max(v1,axis=0)]
        #v1 = (v1-min_max_v1[0])/(min_max_v1[1] - min_max_v1[0])
        #vertices = v1
            #print(self.v_loss)
        #print('vertices.shape:',vertices.shape)
        score_pred = None#(tensor_preds[1][0] * self.corr_mask).cpu()[0].detach().numpy() if self.dict_info['decoder']['do_score'] else None
        return vertices,vertices_mano

    def images_eval(self,image_dir,vis=False,save=True):
        self.model.eval()
        if self.do_super:
            self.model_super.eval()
        if vis:
            mesh, viewer = create_mesh_viewer()
        project=np.array([[480,0,-128,0],[0,-480,-128,0],[0,0,-1,0]],dtype=np.float32).reshape(3,4)
        if args.do_z:
            project = np.array([[256,0,0,0],[0,256,0,0],[0,0,0,1]],dtype=np.float32).reshape(3,4)
        import glob
        datasets = glob.glob(image_dir + '/*.jpg', recursive=True)
        xyz_pred_list=[]
        verts_pred_list=[]
        for fn in datasets:
            img = cv2.imread(fn)
            crop = cv2.resize(img, (256,256))
            vertices = self.base_eval(crop)
            if vis:
                #print(vertices.shape)
                save_obj(fn[:-3]+'obj', vertices, self.faces)
                continue
                uvs = np.matmul(project, np.concatenate((vertices,np.ones_like(vertices[:,-1:])),axis=-1).T).T
                uv=(uvs[:,:-1]/uvs[:,-1:]+0.5).astype(np.int32)
                #print(np.min(uv,axis=0),np.max(uv,axis=0))
                index=[uv[:,1],uv[:,0]]
                index=[np.clip(v, 0, 255) for v in index]
                if args.do_z:
                    x,y,z=[vertices[:,i] for i in range(3)]
                    vertices  = np.stack([x,-y,-z],axis=-1)
                    s=0.5
                else:
                    s=10
                crop[index]=255
                cv2.imshow('hand', crop)
                drawmesh(mesh,vertices*s,viewer)            
                key = cv2.waitKey(1000)
                if key==83 or key ==115 or key==32:
                    #cv2.imwrite(fn.replace('rgb','wa_rgb'),img)
                    save_obj(fn[:-3]+'obj', vertices, self.faces)

                if key == 27:
                    break
            if save:
                joint = np.dot(self.manoJregressor,vertices)
                joint = joint[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
                xyz_pred_list.append(joint)
                verts_pred_list.append(vertices)
                #vertices = np.vstack((vertices,joint))
                save_obj(fn[:-3]+'obj', vertices, self.faces)
            
        if vis:
            cv2.destroyAllWindows()
        if save:
            dump(image_dir+self.model_name[:-3]+'json', xyz_pred_list, verts_pred_list)

    def scale_eval(self, img, d = 20, stage=3):
        if img.shape[0]!=self.size:
            img = cv2.resize(img, (self.size,self.size))   
        height, width, _ = img.shape
        pre_loss=10.
        vertices_= None
        for i in range(stage):
            w = i * d if i else 1
            crop = cv2.resize(img[w:-w,w:-w],(width, height))
            vertices, vertices_mano = self.base_eval(crop)
            if pre_loss>self.sampled_loss:
                pre_loss = self.sampled_loss
                vertices_ = vertices
            #vertices_ = vertices_ + vertices if vertices_ is not None else vertices 
        return vertices_#vertices_/stage
            
    def rotation_eval(self, img, stage=4):
        if img.shape[0]!=self.size:
            img = cv2.resize(img, (self.size,self.size))     
        height, width, _ = img.shape
        vertices_= None 
        pre_loss = 10.  
        v_loss = None
        for i in range(stage):
            M = cv2.getRotationMatrix2D((height//2,width//2),360//stage * i,1)
            crop = cv2.warpAffine(img.copy(),M,(width,height))
            vertices, vertices_mano = self.base_eval(crop)
            if pre_loss>self.sampled_loss:
                pre_loss = self.sampled_loss
                if args.do_mano:
                    v_loss   = self.v_loss
                vertices_ = [vertices,vertices_mano]
            #vertices, vertices_mano = self.base_eval(crop)
            # if vertices_mano is not None:
            #     if pre_loss > self.v_loss:
            #         pre_loss = self.v_loss
            #         vertices_ = [vertices, vertices_mano]
            # else:
            #     if pre_loss>self.sampled_loss:
            #         pre_loss = self.sampled_loss
            #         vertices_ = vertices
            #vertices_ = vertices_ + vertices if vertices_ is not None else vertices
        #print(pre_loss)
        if args.do_mano is False or args.do_mano and v_loss < 0.03:
            return vertices_[0]  
        else:
            return vertices_[1]
        return vertices_#vertices_/stage
    
    def shift_eval(self, img, shift = 30, stage=5):
        width = img.shape[0]
        dirs = [[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,-1],[0,1]]
        dirs = dirs[:stage]
        vertices_ = None
        pre_loss=10.
        for i in range(len(dirs)):
            x, y = dirs[i][0] * shift, dirs[i][1] * shift
            bx, ex = [x, width] if x > 0 else  [0, width + x]
            by, ey = [y, width] if y > 0 else [0, width + y]
            crop = np.zeros_like(img)
            crop[by:ey,bx:ex] = img[by:ey,bx:ex]
            vertices, vertices_mano = self.base_eval(crop)
            if pre_loss>self.sampled_loss:
                pre_loss = self.sampled_loss
                vertices_ = vertices
            #vertices_ = vertices_ + vertices if vertices_ is not None else vertices 
        return vertices_#vertices_/stage

    def rotation_scale_eval(self,img,stage=4,d=25):
        if img.shape[0]!=self.size:
            img = cv2.resize(img, (self.size,self.size))     
        height, width, _ = img.shape
        vertices_= None 
        pre_loss = 10.  
        
        for i in range(stage):
            M = cv2.getRotationMatrix2D((height//2,width//2),90*i,1)
            crop = cv2.warpAffine(img.copy(),M,(width,height))
            for j in range(2):
                w = j * d if j else 1
                crop_ = cv2.resize(crop[w:-w,w:-w],(width, height))
                vertices, vertices_mano = self.base_eval(crop_)
                if pre_loss>self.sampled_loss:
                    pre_loss = self.sampled_loss
                    vertices_ = vertices
            #vertices_ = vertices_ + vertices if vertices_ is not None else vertices 
        return vertices_#vertices_/stage


    def color_eval(self, img):
        vertices1, loss1 = self.base_eval(img),self.sampled_loss
        vertices2, loss2 = self.base_eval(img,False),self.sampled_loss
        return vertices1 if loss1<=loss2 else vertices2

    def base_eval(self, img, rgb=True):
        if img.shape[0]!=self.size:
            img = cv2.resize(img, (self.size,self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
        vertices, vertices_mano =  self.eval(np.array(img))

        return [vertices, vertices_mano]


    def images_z_eval(self,image_dir,vis=False,save=True):
        self.model.eval()
        if self.do_super:
            self.model_super.eval()
        project = np.array([[self.size,0,0,0],[0,self.size,0,0],[0,0,0,1]],dtype=np.float32).reshape(3,4)  if args.do_z else \
                  np.array([[480,0,-128,0],[0,-480,-128,0],[0,0,-1,0]],dtype=np.float32).reshape(3,4)
        import glob
        datasets = glob.glob(image_dir + '/*.jpg', recursive=True)
        xyz_pred_list=[]
        verts_pred_list=[]
        if vis:
            mesh, viewer = create_mesh_viewer()
        for fn in datasets:
            img = cv2.imread(fn)  
            img = cv2.resize(img, (self.size,self.size))          
            
            vertices = self.rotation_eval(img)
            #vertices = self.scale_eval(img)
            #vertices =  self.base_eval(img) if vis else self.rotation_eval(img)
            if type(vertices) is list and vertices[1] is None:
                vertices=vertices[0]
            if vis:
                if args.do_mano and type(vertices) is dict:
                    joint = vertices['joints']
                    vertices = vertices['vertices']
                s = 10
                if args.do_z:
                    x,y,z=[vertices[:,i] for i in range(3)]
                    vertices  = np.stack([x,-y,-z],axis=-1)
                    s=0.2
                if args.do_mano:
                    s=1
                cv2.imshow('hand', img)
                drawmesh(mesh,vertices*s,viewer)            
                key = cv2.waitKey(1000)
                if key==83 or key ==115 or key==32:
                    #cv2.imwrite(fn.replace('rgb','wa_rgb'),img)
                    save_obj(fn[:-3]+'obj', vertices, self.faces)
                if key == 27:
                    break
                
            if save:
                if args.do_mano and type(vertices) is dict:
                    joint = vertices['joints']
                    vertices = vertices['vertices']
                else:
                    joint = np.dot(self.manoJregressor,vertices)
                    joint = joint[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
                #xyz = np.vstack((vertices,joint))
                save_obj(fn[:-3]+'obj', vertices, self.faces)
                xyz_pred_list.append(joint)
                verts_pred_list.append(vertices)
        if save:
            dump(image_dir+self.model_name[:-4]+'.json', xyz_pred_list, verts_pred_list)
    

    def video_eval(self, ip): 
        self.model.eval()
        if self.do_super:
            self.model_super.eval()
        capture = cv2.VideoCapture(ip)
        cnt = 0
        mesh, viewer = create_mesh_viewer()
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        
        project=np.array([[480,0,-128,0],[0,-480,-128,0],[0,0,-1,0]],dtype=np.float32).reshape(3,4)
        if args.do_z:
            project = np.array([[self.size,0,0,0],[0,self.size,0,0],[0,0,0,1]],dtype=np.float32).reshape(3,4)
        index=None
        
        #import matplotlib.pyplot as plt
        #plt.ion()
       
        #plt.figure(figsize=(10, 10))
        #plt.title('result')
        while capture.isOpened():
            hasFrame, frame = capture.read()
            if not hasFrame :
                break
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            crop, box = image_resize(frame)
            #crop = cv2.flip(crop,1)
           
            if cnt %5 == 0:
                vertices, score_pred =  self.eval(np.array(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2,0,1)))
              
                uvs = np.matmul(project, np.concatenate((vertices,np.ones_like(vertices[:,-1:])),axis=-1).T).T
                uv=(uvs[:,:-1]/uvs[:,-1:]+0.5).astype(np.int32)
                #print(np.min(uv,axis=0),np.max(uv,axis=0))
                index=[uv[:,1],uv[:,0]]
                index=[np.clip(v, 0, 255) for v in index]
                
                s = 0.5 if args.do_z else 10
                if args.do_z:
                    x,y,z=[vertices[:,i] for i in range(3)]
                    vertices  = np.stack([x,-y,-z],axis=-1)
                drawmesh(mesh,vertices*s,viewer)
 
            if index:
                crop[index]=255
            cv2.imshow('hand', crop)
            cnt = cnt + 1
            key = cv2.waitKey(20)
            if key == 27:
                break

        #capture.release()
        cv2.destroyAllWindows()
        



   
    def get_results(self, mesh_dir):        
        import glob
        datasets = glob.glob(mesh_dir + '/*.obj', recursive=True)
        xyz_pred_list=[]
        verts_pred_list=[]
        for fn in datasets:
            vertices = read_obj(fn)
            joint = np.dot(self.manoJregressor,vertices)
            joint = joint[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
            xyz_pred_list.append(joint)
            verts_pred_list.append(vertices)
        dump(mesh_dir+'.json', xyz_pred_list, verts_pred_list)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    tester = HandMeshTester(256)
    ip='http://10.7.193.232:8081'#'127.0.0.1'#
    tester.video_eval(ip)
   

