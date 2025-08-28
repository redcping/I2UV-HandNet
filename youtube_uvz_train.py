# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter
import config 
from models import Model,GridMask
import metrics
import numpy as np
import youtube_datasets as datasets
import torch.nn.functional as F
import utils
import argparse
import random
from EMA import EMA

parser = argparse.ArgumentParser()
# for jarvis
parser.add_argument('--data_dir',                type=str,  help='Directory for storing input data')
parser.add_argument('--output_dir',              type=str,  help='Directory for storing output data')
parser.add_argument('--model_dir',               type=str,  help='Directory for storing model')
parser.add_argument('--previous_job_output_dir', type=str,  help='Directory for previous_job_output_dir', default='')
#for user
parser.add_argument('--encoder_name',       type=str,  help="encode net",                    default='resnet50')
parser.add_argument('--datanames',          type=str,  help='datanames',                     default='syth')
parser.add_argument('--load_path',          type=str,  help='load path',                     default='hand.pth')
parser.add_argument('--init_lr',            type=float,help='init_lr',                       default = 0.0001)
parser.add_argument('--img_size',           type=int,  help="image size",                    default=256)
parser.add_argument('--batch_size',         type=int,  help="batch size",                    default=64)
parser.add_argument('--epoch',              type=int,  help="num epoch",                     default=100)
parser.add_argument('--num_workers',        type=int,  help="num workers",                   default=10)
parser.add_argument('--do_uvz',             type=bool, help="do uvz",                        default=True)
parser.add_argument('--do_score',           type=bool, help="do score",                      default=False)
parser.add_argument('--encoder_pretrained', type=bool, help="encode net encoder_pretrained", default=False)
parser.add_argument('--use_multiple_gpu',   type=bool, help="use_multiple_gpu",              default=True)
parser.add_argument('--do_sat',             type=bool, help="do sat",                        default=False)
parser.add_argument('--do_mano',            type=bool, help="do mano",                       default=False)
parser.add_argument('--do_gan',             type=bool, help="do gan",                        default=False)
parser.add_argument('--do_multi_scale',     type=bool, help="do multi-scale",        default=False)
parser.add_argument('--do_z',               type=bool, help="do z",        default=True)
parser.add_argument('--do_mano_cascade',               type=bool, help="do_mano_cascade",        default=False)

args = parser.parse_args()
device_ids=[0,1,2,3] if args.use_multiple_gpu else [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in device_ids])
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class HandMeshTrainer(object):
    '''hand mesh training schedule'''
    def __init__(self):
        super().__init__()
        self.args = args
        print(args.data_dir,args.output_dir,args.model_dir)
        #init dataset
        data_root =  args.data_dir#os.environ['DATASET'] for jarvis
        previous_job_output_dir  = args.previous_job_output_dir
        previous_job_output_dir = previous_job_output_dir.replace('/output/','/model/')
        self.load_path = previous_job_output_dir + '/' + args.load_path if previous_job_output_dir!='' else data_root+'/'+'hand/models/'+args.load_path
        print('load_path:',self.load_path)
        bg_root = os.path.join(data_root,'ocr/data_gen/val_large/') if data_root=='/data/' else os.path.join(data_root,'hand/background') 
        train_data = datasets.ManoData(root=f'{self.args.data_dir}', datanames=f'{self.args.datanames}',
                                                 mode ='train',
                                                 size=self.args.img_size,
                                                 bg_root = bg_root,
                                                 image_transform=datasets.ToTensorTransform(), 
                                                 uvz_transform=datasets.ToTensorTransform())
        val_datanames = 'obman,youtube' 
        val_data = datasets.ManoData(root=f'{self.args.data_dir}', datanames=val_datanames,
                                                 mode ='test',
                                                 size=self.args.img_size,
                                                 bg_root = None,
                                                 image_transform=datasets.ToTensorTransform(), 
                                                 uvz_transform=datasets.ToTensorTransform())
        
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, 
                                                             shuffle=True, num_workers=self.args.num_workers*2, pin_memory=True, drop_last=True)
        
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.args.batch_size, 
                                                             shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)

        self.val_loader_iter = iter(self.val_loader)
        self.size = self.args.img_size 
        #init model
        v_to_vt,faces = utils._parse_obj('./HandRightUV_3K.obj')
        v_to_vt = (v_to_vt * (self.size-1)+0.5).astype(np.int32)
        v_to_vt = v_to_vt.astype(np.float32)/(self.size-1)
        self.v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.v_to_vt = 2 * torch.cat((self.v_to_vt[...,1:2], self.v_to_vt[...,0:1]),-1) - 1

        self.faces = torch.from_numpy(faces.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()

        camera_npz = np.load('./camera_info.npz')
        self.camera_intern = torch.from_numpy(np.concatenate((camera_npz['camera_calib'], np.array([0.,0.,0.]).reshape(3,1)),axis=-1).astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.camera_extern = torch.from_numpy(np.concatenate((camera_npz['camera_extrn'], np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0).astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        corr_mask = np.sum(np.asarray(Image.open('./HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1,keepdims=True) > 0
        corr_mask = corr_mask.astype(np.float32)[None,...].transpose(0,3,1,2)
        self.corr_mask = torch.from_numpy(corr_mask.repeat(self.args.batch_size, axis=0)).cuda().float()
        self.corr_mask = torch.nn.functional.interpolate(self.corr_mask, scale_factor=self.size/256, mode='bilinear', align_corners=True)
        self.GridMask = GridMask()
        self.setup_models()
        self.lr = self.args.init_lr
        if self.args.do_gan:
            model_params = list(self.model.module.encoder.parameters())+list(self.model.module.decoder.parameters())
            self.optim = torch.optim.Adam(model_params, lr=self.lr, betas=(0.5,0.999))
            self.dis_optim = torch.optim.Adam(self.model.module.dis_layer.parameters(), lr=self.lr, betas=(0.5,0.999))
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.ema = EMA(self.model, 0.96)
        #self.optim =  torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,weight_decay=1e-5)
        self.load_pretrain()
        self.total_step=-1
        self.max_iter_step = len(self.train_loader)* self.args.epoch
        print('max_iter:',self.max_iter_step)
        
        #init loss
        self.init_loss()
        
        self.summary_writer = SummaryWriter(self.args.output_dir)


    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 2 every 10 epochs after 20 epoches"""
        self.lr = max(self.args.init_lr * (1. + np.cos(self.total_step * np.pi / self.max_iter_step)),2e-6)*0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def init_loss(self):
        self.laplace_loss_fn = metrics.laplace_loss
        self.mse_loss_fn = metrics.mse_loss
        self.reg_loss_fn =  metrics.reg_loss
        self.smoothness_loss_fn = metrics.smoothness_loss
        self.l1_loss_fn =  metrics.l1_loss
        self.point_loss_fn = torch.nn.L1Loss()
        
        #loss average
        self.syth_losses = AverageMeter('SythnLosses')
        self.real_losses = AverageMeter('RealLosses')
        self.gan_losses = AverageMeter('GANLosses')       


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
        return (uvz-min_v)/(max_v - min_v + 1e-8)*mask
        
    def mask_uvz(self, uvz, mask):
        scale = (1.0 * uvz.size()[-1])/mask.size()[-1]
        mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
        return uvz * mask


    def vis_mask(self,uvz,nums=1):
        mask_maps=[]
        for i in range(nums):
            if uvz.shape[1]>1:
                uv = uvz[i,:2,:,:].permute(1,2,0).cpu().detach().numpy()
            else:
                uv = uvz[i,:,:,:2].cpu().detach().numpy()
            uv = np.array(uv*self.size).astype(np.int32)
            mask = np.zeros((self.size,self.size))
            index = (np.clip(uv[...,1],0,self.size-1), np.clip(uv[...,0], 0, self.size-1))
            mask[index] = 1
            mask = torch.from_numpy(mask.astype(np.float32)[None,...,None]).cuda()
            mask_maps.append(mask)
        maps = torch.cat(mask_maps,0).permute(0,3,1,2)
        return maps


    def joints_dist_map_loss(self, vertices, vertices_gt, sampled_k = 256):
        vertices = vertices[:,:778]
        vertices_gt = vertices_gt[:,:778]
        index = random.sample(range(0, vertices.size()[1]), min(sampled_k,vertices.size()[1]))
        v_sampled, v_gt_sampled = vertices[:,index], vertices_gt[:,index]
        dist_map    = torch.matmul(v_sampled,v_sampled.permute(0,2,1))
        dist_map_gt = torch.matmul(v_gt_sampled,v_gt_sampled.permute(0,2,1))
        delta = dist_map - dist_map_gt
        loss  = torch.sqrt(delta**2+1e-8)
        return torch.mean(loss)

    def self_focal_loss(self, predict, target, stage, flag):
        delta = target - predict
        delta = torch.sqrt(delta.reshape(delta.size()[0], -1)**2 + 1e-12)
         
        pos_mask = (delta < self.smooth_mean).float()
        neg_mask = (delta > self.smooth_mean).float()

        
        prob = 1. - (delta-self.smooth_min)/(self.smooth_max-self.smooth_min + 1e-8)
        prob = torch.clamp(prob, 1e-6, 1.-1e-6)

        pos_loss = -0.25 * torch.pow(torch.sub(1.0, prob), 2) * torch.log(prob) * pos_mask*flag
        neg_loss = -0.75 * torch.pow(prob, 2) * torch.log(torch.sub(1.0, prob)) * neg_mask*flag
        
        pos_loss = torch.sum(pos_loss,dim=-1)/(torch.sum(pos_mask,dim=-1)+1e-8)
        neg_loss = torch.sum(neg_loss,dim=-1)/(torch.sum(neg_mask,dim=-1)+1e-8)
        
        return torch.mean(pos_loss),torch.mean(neg_loss)


    def sample_vertices(self, uvz, k=9):
        dirs = [[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,-1],[0,1]]
        k=min(k,len(dirs))
        dirs = dirs[:k]
        
        b, n, max_p, c = self.v_to_vt.size()
        v_to_vt = self.v_to_vt.repeat([1,1,k,1])
        dirs = [4*torch.tensor(np.array(d,dtype=np.float32).reshape(1,1,1,2),dtype=torch.float32).cuda()/(self.size-1) for d in dirs]
        dirs = [d.repeat([b, n, max_p, 1]) for d in dirs]
        dirs = torch.cat(dirs,dim=2) 
        v_to_vt_ = torch.clamp(v_to_vt + dirs,-1,1)
        vertices = torch.nn.functional.grid_sample(uvz, v_to_vt_)
        flags    = torch.nn.functional.grid_sample(self.corr_mask, v_to_vt_)
        flags    = (flags>0.8).float()
        vertices = vertices * flags
        vertices = torch.sum(vertices,dim=-1,keepdim=False)/torch.sum(flags,dim=-1,keepdim=False)
        vertices = vertices.permute(0, 2, 1)

        return vertices

    def run_stage_loss(self, uvz, uvz_gt, vertices_gt, stage_info, project_mat_gt, stage=0, mode='Train'):
        b, c, h, w  = uvz_gt.size()
        #uvz_src=uvz
        uvz         = torch.nn.functional.interpolate(uvz, scale_factor=w/uvz.size()[-1], mode='bilinear', align_corners=True)*self.corr_mask
       
        flag        = stage_info['flag'].view(-1,1,1,1)
        grid_gt     = stage_info['grid']
        grid_flag   = stage_info['grid_flag'].view(-1,1,1,1)
        min_max_v_gt= stage_info['min_max_v_gt']
        uvz_gt_grid = stage_info['uvz_gt_grid']
        
        z_shift =  torch.sum((uvz_gt[:,-1:,...] - uvz[:,-1:,...]).view(b,1,-1),dim=-1)/torch.sum(self.corr_mask.view(b,1,-1),dim=-1)
        z_shift = z_shift.view(b,1,1,1)

        xy, z       = torch.split(uvz,[2,1],dim=1)
        xy_gt, z_gt = torch.split(uvz_gt,[2,1],dim=1)
        z_shift =  torch.sum((z_gt - z).view(b,1,-1),dim=-1)/torch.sum(self.corr_mask.view(b,1,-1),dim=-1)
        z_shift = z_shift.view(b,1,1,1)
        z = (z + z_shift)*self.corr_mask
       
        '''uvz loss'''
        ''' project loss'''
        uvz_grid_pred = xy 
        flag = grid_flag*flag
        project_loss  = self.l1_loss_fn(xy*flag, xy_gt*flag)
        z_loss  = self.l1_loss_fn(z*flag, z_gt*flag)
        uvz_loss = project_loss + z_loss
        
        '''point_loss'''
        uvz = torch.cat([xy,z],dim=1)*self.corr_mask
        #vertices    = utils.sample_uv_xyz(self.v_to_vt, uvz)
        vertices = self.sample_vertices(uvz)
        point_loss   = self.point_loss_fn(vertices * flag.view(-1,1,1), vertices_gt * flag.view(-1,1,1))
        src_loss = uvz_loss + point_loss

        
        # '''uvz_norm_loss'''
        min_max_v     = min_max_v_gt
        #min_max_v    = [torch.min(vertices,dim=1)[0].view(b,1,3),torch.max(vertices,dim=1)[0].view(b,1,3)]
        #uvz_gt_norm   = self.norm_uvz(uvz_gt, self.corr_mask, [v.view(-1,max(v.shape[1:]),1,1) for v in min_max_v_gt])
        #uvz_norm      = self.norm_uvz(uvz, self.corr_mask, [v.view(-1,max(v.shape[1:]),1,1) for v in min_max_v])
        #uvz_norm_loss = self.l1_loss_fn(uvz_norm*flag, uvz_gt_norm*flag)
        #vertices_gt_norm =  (vertices_gt-min_max_v_gt[0])/(min_max_v_gt[1] - min_max_v_gt[0]+1e-8)
        #vertices_norm    =  (vertices-min_max_v[0])/(min_max_v[1] - min_max_v[0]+1e-8)
        #point_norm_loss  = self.point_loss_fn(vertices_norm * flag.view(-1,1,1), vertices_gt_norm * flag.view(-1,1,1))
        #norm_loss = uvz_norm_loss + point_norm_loss
        #self.summary_loss['{}/uvz_norm_loss_{}'.format(mode,stage)]     = uvz_norm_loss.item()
        #self.summary_loss['{}/point_norm_loss_{}'.format(mode,stage)]   = point_norm_loss.item()
        
        '''smooth_loss'''
        smooth_loss = self.smoothness_loss_fn(uvz*flag, uvz_gt*flag, self.corr_mask)

        '''sampled_loss'''
        sampled_gt   = torch.nn.functional.grid_sample(uvz_gt, self.v_to_vt)
        sampled_pred = torch.nn.functional.grid_sample(uvz,    self.v_to_vt)
        sampled_mean = torch.mean(sampled_pred, dim=-1, keepdim=True).repeat(1,1,1,sampled_gt.size()[-1])
        sampled_loss = self.l1_loss_fn(sampled_pred * flag, sampled_gt*flag) \
                     + self.l1_loss_fn(sampled_mean * flag, sampled_gt*flag)
        
        '''joints_dist_map_loss'''
        #dist_map_loss = self.joints_dist_map_loss(vertices*flag.view(-1,1,1), vertices_gt*flag.view(-1,1,1), sampled_k = 256)
        #self.summary_loss['{}/dist_map_loss_{}'.format(mode,stage)] = dist_map_loss.item() 

        stage_loss =  src_loss + sampled_loss + smooth_loss #+ 0.1*dist_map_loss
        
        if mode=='Tr1ain':
            if stage==0:
                min_max_v   = [torch.min(vertices,dim=1)[0].view(b,1,3),torch.max(vertices,dim=1)[0].view(b,1,3)]
                min_v,max_v = [torch.mean(torch.abs(min_max_v[i]-min_max_v_gt[i])).view(1,-1).detach() for i in range(2)]
                mean = 0.5 * (min_v+max_v)
                step = self.total_step * self.stage + stage
                self.smooth_mean = ((self.smooth_mean * step + mean)/(step + 1)).detach()
                self.smooth_max  = ((self.smooth_max * step + max_v)/(step + 1)).detach()
                self.smooth_min  = ((self.smooth_min * step + min_v)/(step + 1)).detach()
                self.summary_loss['{}/smooth_mean'.format(mode)] = self.smooth_mean.item()
                self.summary_loss['{}/smooth_min'.format(mode)]  = self.smooth_min.item()
                self.summary_loss['{}/smooth_max'.format(mode)]  = self.smooth_max.item()

                uvz_pos_loss, uvz_neg_loss = self.self_focal_loss(uvz, uvz_gt, stage, flag.view(-1,1))
                    #v_pos_loss,   v_neg_loss   = self.self_focal_loss(vertices, vertices_gt, stage, flag.view(-1,1)) 
                pos_loss = uvz_pos_loss #+ v_pos_loss
                neg_loss = uvz_neg_loss #+ v_neg_loss
                focal_loss = pos_loss + neg_loss
                stage_loss = stage_loss + 0.2*focal_loss
                self.summary_loss['{}/focal_loss_{}'.format(mode,stage)] = focal_loss.item()
                self.summary_loss['{}/pos_loss_{}'.format(mode,stage)] = pos_loss.item()
                self.summary_loss['{}/neg_loss_{}'.format(mode,stage)] = neg_loss.item()
        
        '''summary loss'''
        k=1
        self.summary_loss['{}/uvz_loss_{}'.format(mode,stage)]     = uvz_loss.item()
        self.summary_loss['{}/project_loss_{}'.format(mode,stage)] = project_loss.item()
        self.summary_loss['{}/z_loss_{}'.format(mode,stage)] = z_loss.item()
        #self.summary_loss['{}/ no_uvz_project_loss_{}'.format(mode,stage)] =  no_uvz_project_loss.item()
        self.summary_loss['{}/point_loss_{}'.format(mode,stage)]   = point_loss.item()
        #self.summary_loss['{}/smooth_loss_{}'.format(mode,stage)]  = smooth_loss.item()
        self.summary_loss['{}/sampled_loss_{}'.format(mode,stage)] = sampled_loss.item()
        self.summary_loss['{}/stage_loss_{}'.format(mode,stage)]   = stage_loss.item()

       
        self.summary_image['{}/uvz_pred_{}'.format(mode,stage)]  = self.norm_uvz(uvz,self.corr_mask,min_max_v_gt)[:k]
        self.summary_image['{}/mask_pred_{}'.format(mode,stage)] = self.vis_mask(uvz_grid_pred[:k])
    

        '''summary vertices'''
        if stage < 2:
            vertices_ = torch.stack([vertices[...,0],-vertices[...,1],-vertices[...,2]],dim=-1)
            self.summary_vertices['{}_vertices_{}'.format(mode,stage)] = vertices_[:k]
            
        '''summary histogram'''     
        for j in range(3):
            self.summary_histogram['{}/{}_v_pred_{}'.format(mode,stage,j)] = vertices[...,j]
        self.summary_histogram['{}/z_shift_{}'.format(mode,stage)] = z_shift
                
        return stage_loss
    
    def run_loss(self, uvz_preds, uvz_gt, image_gt, project_info, project_mat_gt, mode='Train'):
        b, c, h, w  = uvz_gt.size()
        uvz_preds   = uvz_preds[:self.stage] 
        #vertices_gt = utils.sample_uv_xyz(self.v_to_vt, uvz_gt)
        vertices_gt = self.sample_vertices(uvz_gt)
        min_max_v_gt= [torch.min(vertices_gt,dim=1)[0].view(b,1,3),torch.max(vertices_gt,dim=1)[0].view(b,1,3)]
        flag        = project_info['flag'].view(-1,1,1,1)
        grid_gt     = project_info['grid']
        grid_flag   = project_info['grid_flag'].view(-1,1,1,1)
        uvz_gt_grid = torch.clamp(utils.project_map(uvz_gt, project_mat_gt, 1.),0.,1.) * self.corr_mask
        stage_info  = dict(flag=flag,grid_flag=grid_flag,grid=grid_gt,min_max_v_gt=min_max_v_gt,uvz_gt_grid=uvz_gt_grid)
        
        '''summary gt'''
        self.summary_vertices['{}_vertices_gt'.format(mode)] = torch.stack([vertices_gt[...,0],-vertices_gt[...,1],-vertices_gt[...,2]],dim=-1)
        for j in range(3):
            self.summary_histogram['{}/v_gt_{}'.format(mode,j)] = vertices_gt[...,j]
        self.summary_image['{}/image_gt'.format(mode)]   = image_gt
        self.summary_image['{}/uvz_gt'.format(mode)] = uvz_gt
        self.summary_image['{}/mask_gt'.format(mode)] = project_info['mask']
        self.summary_image['{}/mask_gt_uvz'.format(mode)] = self.vis_mask(uvz_gt_grid[:1])
        
        stages_loss = [self.run_stage_loss(uvz_preds[i], uvz_gt, vertices_gt, stage_info,project_mat_gt,stage=i,mode=mode) for i in range(self.stage)]
        #stages_sampled_loss = [losses[i]['sampled_loss'] for i in range(self.stage)]
        #self.stages_loss = torch.stack([losses[i]['stage_loss'] for i in range(self.stage)],)
        self.stages_loss = torch.stack(stages_loss, dim=0)
        min_stage        = torch.min(self.stages_loss,dim=0)[1]
        self.stages_loss = torch.mean(self.stages_loss)
        self.select_loss = stages_loss[min_stage.unsqueeze(0)]
        self.total_loss  = self.stages_loss #if np.random.randint(0, 2)>0 else self.select_loss
        self.summary_loss['{}/stages_loss'.format(mode)] = self.stages_loss.item()
        self.summary_loss['{}/select_loss'.format(mode)] = self.select_loss.item()         
        self.summary_loss['{}/total_loss'.format(mode)] = self.total_loss.item()
        self.summary_loss['{}/min_stage'.format(mode)] = min_stage.item()
        

        '''total loss''' 
        #self.total_loss +=   self.uvz_loss +  0.5 * self.point_loss + 0.1 * self.project_loss +  self.smooth_loss + 0.1 * self.sampled_loss

        '''mano point loss'''
        if self.args.do_mano:
            mano_vertices = self.mano['vertices']
            mano_vertices = torch.stack([mano_vertices[...,0],-mano_vertices[...,1],-mano_vertices[...,2]],dim=-1)

            min_max_v_mano = [torch.min(mano_vertices,dim=1)[0].view(b,1,3),torch.max(mano_vertices,dim=1)[0].view(b,1,3)]
            mano_vertices_norm = (mano_vertices - min_max_v_mano[0])/(min_max_v_mano[1] - min_max_v_mano[0] + 1e-8)
            vertices_gt_norm =  (vertices_gt-min_max_v_gt[0])/(min_max_v_gt[1] - min_max_v_gt[0]+1e-8)

            self.mano_point_loss = self.point_loss_fn(mano_vertices_norm*flag, vertices_gt_norm[:,:778]*flag)
            self.total_loss = self.total_loss + self.mano_point_loss
            mano_vertices_ = torch.stack([mano_vertices_norm[...,0],-mano_vertices_norm[...,1],-mano_vertices_norm[...,2]],dim=-1)
            self.summary_vertices['{}/vertices_mano'.format(mode)]   = mano_vertices_
            self.summary_loss['{}/mano_point_loss'.format(mode)] = self.mano_point_loss
            self.summary_histogram['{}/v_mano'.format(mode)] = self.mano['vertices']

        


    
    def run_adversarial_loss(self,adversarial,mode='Train'):
        adversarial_loss = torch.nn.MSELoss()
        real_x = adversarial['real_x']
        gen_x = adversarial['gen_x']
        gen_x_detach = adversarial['gen_x_detach']
        
        valid = torch.autograd.Variable(torch.cuda.FloatTensor(real_x.size(0), 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_x.size(0), 1).fill_(0.0), requires_grad=False)
        
        self.g_loss = adversarial_loss(gen_x, valid)
        
        real_loss = adversarial_loss(real_x, valid)
        fake_loss = adversarial_loss(gen_x_detach, fake)
        self.dis_loss = 0.5 * (real_loss + fake_loss)
        
        self.total_loss = self.total_loss+ 0.05 * self.g_loss
        self.summary_loss['{}/g_loss'.format(mode)]     = self.g_loss
        self.summary_loss['{}/d_loss'.format(mode)]     = self.dis_loss
        self.summary_loss['{}/total_loss'.format(mode)] = self.total_loss
    
    
    def run_mano_decoder_loss(self, outputs, uvz_gt, image_gt, project_info, project_mat_gt, mode='Train'):
        
        b, c, h, w      = uvz_gt.size()        
        vertices_gt     = utils.sample_uv_xyz(self.v_to_vt, uvz_gt)
        min_max_v = [torch.min(vertices_gt,dim=1)[0].view(b,1,3),torch.max(vertices_gt,dim=1)[0].view(b,1,3)]
        vertices_gt = (vertices_gt- min_max_v[0])/(min_max_v[1] - min_max_v[0] + 1e-8)
            
        vertices_stages = outputs['vertices_stages']
        stage = len(vertices_stages) if mode=='Train' else 1
        joints_stages   = outputs['joints_stages']
        flag = project_info['flag'].view(-1,1,1)
        grid_vertices_loss, grid_joints_loss, mano_point_loss = 0,0,0
        grid_gt, grid_flag  = project_info['grid'], project_info['grid_flag'].view(-1,1,1,1)
        for i in range(stage):
            v_to_vt   =  torch.cat((vertices_stages[i][...,1:2], vertices_stages[i][...,0:1]),-1).unsqueeze(1)
            sampled   = torch.nn.functional.grid_sample(grid_gt, 2 * v_to_vt - 1).permute(0, 2, 3, 1)
            gv_loss   = self.point_loss_fn(v_to_vt*grid_flag, sampled*grid_flag)
            grid_vertices_loss = grid_vertices_loss + gv_loss
            self.summary_loss['{}/grid_vertices_loss{}'.format(mode,i)]  = gv_loss
            self.summary_image['{}/mask_pred{}'.format(mode,i)]    = self.vis_mask(sampled)
            
            v_to_vt   =  torch.cat((joints_stages[i][...,1:2], joints_stages[i][...,0:1]),-1).unsqueeze(1)
            sampled   = torch.nn.functional.grid_sample(grid_gt, 2 * v_to_vt - 1).permute(0, 2, 3, 1)
            gj_loss   = self.point_loss_fn(v_to_vt*grid_flag, sampled*grid_flag)
            grid_joints_loss = grid_joints_loss+ gj_loss
            self.summary_loss['{}/grid_joints_loss{}'.format(mode,i)]  = gj_loss
            
            min_max_v = [torch.min(vertices_stages[i],dim=1)[0].view(b,1,3),torch.max(vertices_stages[i],dim=1)[0].view(b,1,3)]
            vertices_stages[i] = (vertices_stages[i] - min_max_v[0])/(min_max_v[1] - min_max_v[0] + 1e-8)
            mp_loss = self.point_loss_fn(vertices_stages[i]*flag, vertices_gt[:,:778]*flag)
            mano_point_loss = mano_point_loss+ mp_loss            
            self.summary_loss['{}/cmano_point_loss{}'.format(mode,i)]  = mp_loss
        
        self.total_loss = mano_point_loss + 0.5 * (grid_vertices_loss + grid_joints_loss)
        
        self.summary_loss['{}/grid_vertices_loss'.format(mode)]  = grid_vertices_loss
        self.summary_loss['{}/grid_joints_loss'.format(mode)]    = grid_joints_loss
        self.summary_loss['{}/cmano_point_loss'.format(mode)]     = mano_point_loss
        self.summary_loss['{}/total_loss'.format(mode)]          = self.total_loss
        
        self.summary_image['{}/image_gt'.format(mode)]       = image_gt       
        self.summary_image['{}/mask'.format(mode)]    = project_info['mask']
        
        
        '''summary vertices'''
        vertices = torch.stack([vertices_stages[0][...,0],-vertices_stages[0][...,1],-vertices_stages[0][...,2]],dim=-1).detach()
        vertices_gt = torch.stack([vertices_gt[...,0],-vertices_gt[...,1],-vertices_gt[...,2]],dim=-1)
        self.summary_vertices['{}/vertices'.format(mode)]    = vertices
        self.summary_vertices['{}/vertices_gt'.format(mode)] = vertices_gt

        self.summary_histogram['{}/pose_shape_trans'.format(mode)]    = outputs['pose_shape_trans']
        self.summary_histogram['{}/joints'.format(mode)]    = joints_stages[0]
        
    def run_val_batch(self):
        mode = 'Val'
        try:
            image_gt, uvz_gt, project_info = next(self.val_loader_iter)
        except:
             self.val_loader_iter = iter(self.val_loader)
             image_gt, uvz_gt, project_info = next(self.val_loader_iter)
        project_mat_gt = self.project_mat.cuda(non_blocking=True)
        image_gt = image_gt.cuda(non_blocking=True)
        project_info = {k:v.cuda(non_blocking=True) for k,v in project_info.items()}
        #resize_ratio = resize_ratio.cuda(non_blocking=True).float()
        uvz_gt = uvz_gt.cuda(non_blocking=True)*self.corr_mask

        scale = 1. + 32 * np.random.randint(-1,2) / self.size if self.args.do_multi_scale else 1.
        self.run_info['image'] = torch.nn.functional.interpolate(image_gt, scale_factor=scale, mode='bilinear', align_corners=True)
        self.run_info['project_mat'] = torch.cat((project_mat_gt[...,:-1] * scale, project_mat_gt[...,-1:]),dim=-1)       
         
        if 'real' in self.run_info.keys():
            self.run_info.pop('real')
        outputs = self.model.forward(self.run_info)
        
        self.uvz_preds, self.score_preds=None,None
        if self.dict_info['decoder']['do_uvz'] and self.args.do_mano_cascade is False:
            self.uvz_preds = outputs['uvz_preds']
            
        if self.dict_info['decoder']['do_score']:
            self.score_preds = outputs['score_preds']
            
        if self.args.do_mano:
            self.mano = outputs['mano']

        if self.args.do_mano_cascade:
            self.run_mano_decoder_loss(outputs, uvz_gt, image_gt, project_info, project_mat_gt, mode=mode)
        else:    
            self.run_loss(self.uvz_preds, uvz_gt, image_gt, project_info, project_mat_gt,mode=mode)

    def write_summary(self):
        k=1
        self.summary_writer.add_scalar('Train/lr', self.lr, self.total_step)
        try:
        #if True:
            for key, value in self.summary_vertices.items():
                self.summary_writer.add_mesh('{}'.format(key), utils.normalize_vertices(value[:k]), colors=255*torch.ones(value[:k].shape, dtype=torch.uint8),  faces=self.faces[:k], global_step=self.total_step)

            for key, value in self.summary_image.items():
                self.summary_writer.add_images('{}'.format(key), value[:k], self.total_step)

            for key, value in self.summary_loss.items():
                self.summary_writer.add_scalar('{}'.format(key), value, self.total_step)

            for key, value in self.summary_histogram.items():
                self.summary_writer.add_histogram('{}'.format(key), value[:k], self.total_step)
        except:
            pass
 
 
    def train(self):
        #self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        pre_epoch = 0
        self.run_info=dict(image=None, project_mat=self.project_mat, corr_mask=self.corr_mask, v_to_vt=self.v_to_vt)
        self.real=None 
        self.stage=4
        self.smooth_mean = torch.autograd.Variable(torch.zeros(1,1),requires_grad=False).cuda()
        self.smooth_max  = torch.autograd.Variable(torch.zeros(1,1),requires_grad=False).cuda()
        self.smooth_min  = torch.autograd.Variable(torch.zeros(1,1),requires_grad=False).cuda()
        for epoch in range(pre_epoch,self.args.epoch):
            self.GridMask.set_prob(epoch+self.args.epoch//2,self.args.epoch)
            self.train_epoch(epoch)
            self.save_model(epoch)



    def train_epoch(self, epoch):
        self.begin_train()
        pre_loss=None
        self.project_mat = self.project_mat.detach()
        self.corr_mask = self.corr_mask.detach()
        for n, (image_gt,uvz_gt, project_info) in enumerate(self.train_loader):
            
            self.total_step = self.total_step + 1
            if self.lr>1e-6:
                self.adjust_learning_rate(self.optim)
            if self.args.do_gan:
                self.adjust_learning_rate(self.dis_optim)
            
            b,c,h,w=image_gt.size()
            
            image_gt = image_gt.cuda(non_blocking=True)

            #image_gt = self.GridMask(image_gt) 
            project_info = {k:v.cuda(non_blocking=True) for k,v in project_info.items()}
            uvz_gt = uvz_gt.cuda(non_blocking=True)*self.corr_mask
            project_mat_gt = self.project_mat.cuda(non_blocking=True)
            
            
            scale = 1. + 32 * np.random.randint(-1,2) / self.size if self.args.do_multi_scale else 1.


            self.run_info['image'] = torch.nn.functional.interpolate(image_gt, scale_factor=scale, mode='bilinear', align_corners=True)
            self.run_info['project_mat'] = torch.cat((project_mat_gt[...,:-1] * scale, project_mat_gt[...,-1:]),dim=-1)
            
    
            self.summary_loss={}
            self.summary_image={}
            self.summary_vertices={}
            self.summary_histogram={}

            if self.args.do_gan:
                self.real = uvz_gt if self.real is None or (np.random.randint(0,10) < 5 and self.total_step%7==0) else self.real
                self.run_info.update({'real':self.real}) 
            else:
                self.run_info.update({'real': None})
                self.stage=4
                #self.run_info.update({'real':uvz_gt})
                #if 'real' in self.run_info.keys():
                #    self.run_info.pop('real')
            outputs = self.model.forward(self.run_info)
                
            self.uvz_preds, self.score_preds=None,None
            if self.dict_info['decoder']['do_uvz'] and self.args.do_mano_cascade is False:
                self.uvz_preds = outputs['uvz_preds']
                
            if self.dict_info['decoder']['do_score']:
                self.score_preds = outputs['score_preds']
                
            if self.args.do_mano:
                self.mano = outputs['mano']
            
            if self.args.do_mano_cascade:
                self.run_mano_decoder_loss(outputs, uvz_gt, image_gt, project_info, project_mat_gt, mode='Train')
            else:    
                self.run_loss(self.uvz_preds,uvz_gt, image_gt, project_info, project_mat_gt,'Train')
            

            self.syth_losses.update(self.total_loss, image_gt.shape[0])
            self.summary_loss['Train/avg_loss'] = self.syth_losses.avg
            
           
            if self.args.do_gan:
                adversarial = outputs['adversarial']
                self.run_adversarial_loss(adversarial)
                self.dis_optim.zero_grad()
                self.dis_loss.backward()
                self.dis_optim.step()
            

            self.optim.zero_grad()
            self.total_loss.backward()
            self.optim.step()
            self.ema.update_params()
            #self.total_loss /=4
            #self.total_loss.backward()
            #if n%4==0:
            #    self.optim.step()
            #    self.optim.zero_grad()

            if n % 50 == 0:
                self.model.eval()
                self.ema.apply_shadow()
                self.run_val_batch()
                self.ema.restore()
                self.model.train()
                self.write_summary()
                print('epoch:{},iter:{},total_loss:{}'.format(epoch,self.total_step,self.total_loss.cpu()))
                
    

    def valid_epoch(self):
        self.begin_valid()           

    def begin_train(self):
        self.model.train()
  
    
    def begin_valid(self):
        self.model.eval()
     
    
    def setup_models(self):
        #self.project_mat = torch.matmul(self.camera_intern, self.camera_extern)
        self.project_mat = torch.from_numpy(np.array([[self.size,0,0,0],[0,self.size,0,0],[0,0,0,1]],dtype=np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.dict_info={}
        self.dict_info['encoder'] = {'num_in_layers':3,'encoder_name':args.encoder_name,'pretrained':args.encoder_pretrained}
        self.dict_info['decoder'] = {'do_uvz':args.do_uvz,'do_score':args.do_score,'do_z':self.args.do_z}
        if self.args.do_gan:
            self.dict_info['do_gan'] = self.args.do_gan
            self.dict_info['image_size'] = self.args.img_size
            print('do_gan')
        #self.dict_info['corr_mask'] = self.corr_mask
        if self.args.do_mano:
            self.dict_info['do_mano'] = True
            print('do_mano')
        if self.args.do_mano_cascade:
            self.dict_info['do_mano_cascade'] = True
            self.dict_info['do_mano_cascade_flip'] = False
        print('do_score:', self.dict_info['decoder']['do_score'])
        self.model = Model(self.dict_info)
        if self.args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
      
    def save_model(self, epoch):
        self.ema.apply_shadow()
        state = {
                 'state':self.model.state_dict(),
                 'optimizer':self.optim.state_dict()
                 }
        model_dir = self.args.model_dir
        prefix = 'score_' if self.args.do_score else ''
        torch.save(state, f'{model_dir}/{prefix}hand_recon_parameters_epoch{epoch}.pth')
        self.ema.restore()
        

    def ex_model(self,k,ex=['corr_mask','v_to_vt','camera_intern_extern','project_mat']):
        for e in ex:
            if e in k:
                return False
        return True

    def load_model(self):
        if self.args.encoder_pretrained:
            return
        #checkpoint = torch.load(self.train_cfg.TRAIN.LOAD_PATH)
        model_path = self.load_path
        #'''
        #if self.dict_info['decoder']['do_score']:
        #    model_path = model_path[:-4]+'_score.pth'
        checkpoint = torch.load(model_path)
        pretrained_dict_ = checkpoint['state']
        #import pdb;pdb.set_trace()
        model_dict = self.model.state_dict()
        #import pdb;pdb.set_trace()

        ex = ['corr_mask','v_to_vt','camera_intern_extern','project_mat','mano_decoder','dis_layer','mano_cascade_decoder']
        #ex = ex + ['iconv2','out2_layer','iconv1','out1_layer','upconv1']
        if args.do_mano:
            ex.remove('mano_decoder')
        if args.do_gan:
            ex.remove('dis_layer')
        if args.do_mano_cascade:
            ex.remove('mano_cascade_decoder')
            ex.append('decoder')


        predict='module.' if self.args.use_multiple_gpu else ''

        
        pretrained_dict = {(k if 'module' in k else predict+k):v for k, v in pretrained_dict_.items() if self.ex_model(k,ex)}
        print('pretrain nodes:',len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        #else:
        #'''
        #import pdb;pdb.set_trace()
        #checkpoint = torch.load(model_path)
        #print('len state is :', len(checkpoint['state']))
        self.model.load_state_dict(checkpoint['state'])
        #self.optim.load_state_dict(checkpoint['optimizer'])
        #'''

    def load_pretrain(self):
        print(self.args.load_path)
        self.load_model()
        #if self.train_cfg.TRAIN.LOAD_PATH:
        #    self.model.load_backbone_pretrain(self.train_cfg.TRAIN.LOAD_PATH)


    
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    trainer = HandMeshTrainer()
    trainer.train()

