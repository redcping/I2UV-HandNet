# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter
import config 
from models import Model
import metrics
import numpy as np
import norm_datasets as datasets
import torch.nn.functional as F
import utils
import argparse
import cv2

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
parser.add_argument('--use_multiple_gpu',   type=bool, help="use_multiple_gpu",              default=False)
parser.add_argument('--do_sat',             type=bool, help="do sat",                        default=False)
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
        val_datanames = 'xvx_frei_130k,obman' 
        val_data = datasets.ManoData(root=f'{self.args.data_dir}', datanames=val_datanames,
                                                 mode ='val',
                                                 size=self.args.img_size,
                                                 bg_root = bg_root,
                                                 image_transform=datasets.ToTensorTransform(), 
                                                 uvz_transform=datasets.ToTensorTransform())
        
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, 
                                                             shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
        
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.args.batch_size, 
                                                             shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)

        self.val_loader_iter = iter(self.val_loader)
        self.size = self.args.img_size 
        #init model
        v_to_vt,faces = utils._parse_obj('./HandRightUV.obj')
        v_to_vt = (v_to_vt * (self.size-1)+0.5).astype(np.int32)
        v_to_vt = v_to_vt.astype(np.float32)/(self.size-1)
        self.v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.v_to_vt = 2 * torch.cat((self.v_to_vt[...,1:2], self.v_to_vt[...,0:1]),-1) - 1

        self.faces = torch.from_numpy(faces.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()

        camera_npz = np.load('./camera_info.npz')
        self.camera_intern = torch.from_numpy(np.concatenate((camera_npz['camera_calib'], np.array([0.,0.,0.]).reshape(3,1)),axis=-1).astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.camera_extern = torch.from_numpy(np.concatenate((camera_npz['camera_extrn'], np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0).astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        corr_mask = np.sum(np.asarray(Image.open('./HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1,keepdims=True)>0
        corr_mask = corr_mask[...,:1]
        #corr_mask = cv2.dilate(corr_mask.astype(np.uint8),np.ones((3, 3), np.uint8), iterations=1) 
        corr_mask = corr_mask.astype(np.float32)[None,...].transpose(0,3,1,2)
        print('corr_mask:',corr_mask.shape)
        self.corr_mask = torch.from_numpy(corr_mask.repeat(self.args.batch_size, axis=0)).cuda().float()
        self.corr_mask = torch.nn.functional.interpolate(self.corr_mask, scale_factor=self.size/256, mode='bilinear', align_corners=True)
        self.setup_models()
        self.lr = self.args.init_lr
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.load_pretrain()
        
        #init loss
        self.init_loss()
        
        self.summary_writer = SummaryWriter(self.args.output_dir)


    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 10 epochs after 20 epoches"""
        self.lr = max(self.lr * (0.5**(max(epoch - 20, 0)//10)),1e-6)
        print(epoch,self.lr)
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
        return (uvz-min_v)/(max_v-min_v + 1e-8)*mask
        
    def mask_uvz(self, uvz, mask):
        scale = (1.0 * uvz.size()[-1])/mask.size()[-1]
        mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
        return uvz * mask
    

    def vis_mask(self,uvz):
        mask_maps=[]
        for i in range(1):
            uv = uvz[i,:2,:,:].permute(1,2,0).cpu().detach().numpy()
            uv = np.array(uv*self.size).astype(np.int32)
            mask = np.zeros((self.size,self.size))
            index = (np.clip(uv[...,1],0,self.size-1), np.clip(uv[...,0], 0, self.size-1))
            mask[index] = 1
            mask = torch.from_numpy(mask.astype(np.float32)[None,...,None]).cuda()
            mask_maps.append(mask)
        maps = torch.cat(mask_maps,0).permute(0,3,1,2)
        return maps
 

    def run_loss(self, uvz_preds, uvz_gt, image_gt, project_info, project_mat_gt, mode='Train'):

        stage = 2 if mode=='Train' else 1
        #project_mat_gt = self.model.module.project_mat if self.args.use_multiple_gpu else  self.model.project_mat
        v_to_vt = self.v_to_vt#self.model.module.v_to_vt if self.args.use_multiple_gpu else  self.model.v_to_vt
        corr_mask = self.corr_mask#self.model.module.corr_mask if self.args.use_multiple_gpu else  self.model.corr_mask
        #self.v_to_vt.to(self.uvz_gt.device)
        #self.corr_mask.to(self.uvz_gt.device)
        #project_mat_gt.to(self.uvz_gt.device)
         
        b, c, h, w = uvz_gt.size()
        vertices_gt   = utils.sample_uv_xyz(v_to_vt, uvz_gt)
        min_max_v_gt = [torch.min(vertices_gt,dim=1)[0].view(b,1,3),torch.max(vertices_gt,dim=1)[0].view(b,1,3)]
        vertices = [utils.sample_uv_xyz(v_to_vt, uvz_preds[i]) for i in range(stage)]
        min_max_v_pred = [torch.min(vertices[0],dim=1)[0].view(b,1,3),torch.max(vertices[0],dim=1)[0].view(b,1,3)]
        flag = project_info['crop_r'].view(-1,1,1)
        
        #losses

        ''' project loss'''
        grid_gt, grid_flag  = project_info['grid'], project_info['grid_flag'].view(-1,1,1,1)
        uvz_grid_preds = [torch.clamp(utils.project_map(uvz_preds[i], project_mat_gt, 2**i),0.,1.) for i in range(stage)]
        uvz_grid_preds = [self.mask_uvz(g,corr_mask) for g in uvz_grid_preds]
        uvz_grid_gt = F.grid_sample(grid_gt, 2 * uvz_grid_preds[0].permute(0,2,3,1) - 1, mode='bilinear', padding_mode='zeros') * corr_mask
        
        if self.dict_info['decoder']['do_score']:
            project_loss = [self.laplace_loss_fn(uvz_grid_preds[i]*grid_flag, uvz_grid_gt*grid_flag,self.score_preds[i])/(2**i) for i in range(stage)]
        else:
            project_loss = [self.l1_loss_fn(uvz_grid_preds[i]*grid_flag, uvz_grid_gt*grid_flag)/(2**i) for i in range(stage)]
        self.project_loss = torch.sum(torch.stack(project_loss, dim=0))

        '''uvz_loss'''   
        min_max_v_pred = min_max_v_gt

        uvz_gt_norm = self.norm_uvz(uvz_gt, corr_mask, [v.view(-1,max(v.shape[1:]),1,1) for v in min_max_v_gt])
        uvz_preds_norm = [self.norm_uvz(uvz_preds[i], corr_mask, [v.view(-1,max(v.shape[1:]),1,1) for v in min_max_v_pred]) for i in range(stage)]
        
        norm_flag = project_info['no_youtube'].view(-1,1,1,1)
 
        if self.dict_info['decoder']['do_score']:
            uvz_norm_loss=[self.laplace_loss_fn(uvz_preds_norm[i]*norm_flag, uvz_gt_norm*norm_flag, self.score_preds[i])/(2**i) for i in range(stage)]
            uvz_loss = [self.laplace_loss_fn(uvz_preds[i]*flag.view(-1,1,1,1), uvz_gt*flag.view(-1,1,1,1), self.score_preds[i])/(2**i) for i in range(stage)]
        else:
            uvz_norm_loss = [self.l1_loss_fn(uvz_preds_norm[i]*norm_flag, uvz_gt_norm*norm_flag)/(2**i) for i in range(stage)]
            uvz_loss = [self.l1_loss_fn(uvz_preds[i]*flag.view(-1,1,1,1), uvz_gt*flag.view(-1,1,1,1))/(2**i) for i in range(stage)] 
        self.uvz_norm_loss = torch.sum(torch.stack(uvz_norm_loss, dim=0))
        self.uvz_loss = torch.sum(torch.stack(uvz_loss, dim=0))

        '''points loss'''
        vertices_gt_norm = (vertices_gt - min_max_v_gt[0])/(min_max_v_gt[1] - min_max_v_gt[0])
        vertices_norm = [(vertices[i] - min_max_v_pred[0])/(min_max_v_pred[1] - min_max_v_pred[0]) for i in range(stage)]
        point_norm_loss=[self.point_loss_fn(vertices_norm[i]*norm_flag.view(-1,1,1), vertices_gt_norm*norm_flag.view(-1,1,1)) for i in range(stage)]
        
        point_loss=[self.point_loss_fn(vertices[i]*flag, vertices_gt*flag) for i in range(stage)]
        self.point_loss = torch.sum(torch.stack(point_loss,dim=0))
        self.point_norm_loss = torch.sum(torch.stack(point_norm_loss,dim=0))
        

        '''total_loss'''
        self.total_loss =   0.1 * (self.point_norm_loss + self.uvz_norm_loss) + 2*(self.project_loss + self.uvz_loss +  self.point_loss)

        '''summary loss'''
        self.summary_loss['{}/uvz_loss'.format(mode)]      = uvz_loss[0]
        self.summary_loss['{}/uvz_norm_loss'.format(mode)] = uvz_norm_loss[0]
        self.summary_loss['{}/project_loss'.format(mode)]  = project_loss[0]
        self.summary_loss['{}/point_norm_loss'.format(mode)]  = point_norm_loss[0]
        self.summary_loss['{}/point_loss'.format(mode)]  = point_loss[0]
        self.summary_loss['{}/total_loss'.format(mode)]    = self.total_loss

        '''summary image'''
        self.summary_image['{}/image_gt'.format(mode)]       = image_gt
        self.summary_image['{}/uvz_pred_norm'.format(mode)]  = uvz_preds_norm[0]
        self.summary_image['{}/uvz_gt_norm'.format(mode)]    = uvz_gt_norm
        self.summary_image['{}/mask'.format(mode)]    = project_info['mask']
        self.summary_image['{}/mask_pred'.format(mode)]    = self.vis_mask(uvz_grid_preds[0])
        
        if self.dict_info['decoder']['do_score']:
            score = 3.0 * torch.tanh(self.score_preds[0] / 3.0)
            self.summary_image['{}/scores'.format(mode)] = score
            self.summary_histogram['{}/score'.format(mode)]  = score

        '''summary vertices'''
        self.summary_vertices['{}/vertices'.format(mode)]    = vertices[0]
        self.summary_vertices['{}/vertices_gt'.format(mode)] = vertices_gt

        '''summary histogram'''
        self.summary_histogram['{}/uvz'.format(mode)]    = uvz_preds[0]
        self.summary_histogram['{}/uvz_gt'.format(mode)] = uvz_gt
        
    def run_val_batch(self):
        mode = 'Val'
        try:
            image_gt, uvz_gt, project_info = next(self.val_loader_iter)
        except:
             self.val_loader_iter = iter(self.val_loader)
             image_gt, uvz_gt, project_info = next(self.val_loader_iter)
        project_mat_gt = self.project_mat 
        image_gt = image_gt.cuda(non_blocking=True)
        project_info = {k:v.cuda(non_blocking=True) for k,v in project_info.items()}
        #resize_ratio = resize_ratio.cuda(non_blocking=True).float()
        uvz_gt = uvz_gt.cuda(non_blocking=True)
        #project_mat_gt = project_mat_gt(non_blocking=True)#project_mat_gt.to(uvz_gt.device)
        tensor_preds = self.model.forward(image_gt)
        
        self.uvz_preds, self.score_preds=None,None
        if self.dict_info['decoder']['do_uvz']:
            self.uvz_preds = [self.mask_uvz(tensor[0],self.corr_mask) for tensor in tensor_preds[:4]]
            uvz_gt         = self.mask_uvz(uvz_gt,self.corr_mask)
        if self.dict_info['decoder']['do_score']:
            self.score_preds = [self.mask_uvz(tensor[1],self.corr_mask) for tensor in tensor_preds[:4]]
            
        
        self.run_loss(self.uvz_preds, uvz_gt, image_gt, project_info, project_mat_gt, mode=mode)

    def write_summary(self):
        k=1
        self.summary_writer.add_scalar('Train/lr', self.lr, self.total_step)
        try:
            for key, value in self.summary_vertices.items():
                self.summary_writer.add_mesh('{}'.format(key), utils.normalize_vertices(value[:k]), colors=255*torch.ones(value[:k].shape, dtype=torch.uint8),  faces=self.faces[:k], global_step=self.total_step)

            for key, value in self.summary_image.items():
                self.summary_writer.add_images('{}'.format(key), value[:k], self.total_step)

            for key, value in self.summary_loss.items():
                self.summary_writer.add_scalar('{}'.format(key), value.item(), self.total_step)

            for key, value in self.summary_histogram.items():
                self.summary_writer.add_histogram('{}'.format(key), value[:k], self.total_step)
        except:
            pass
 
 
    def train(self):
        #self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.total_step=0
        pre_epoch = 0
        for epoch in range(pre_epoch,self.args.epoch):
            self.adjust_learning_rate(self.optim, epoch)
            self.train_epoch(epoch)
            self.save_model(epoch)

    def train_epoch(self, epoch):
        self.begin_train()
        max_iter_per_epoch = len(self.train_loader)//100 * 90
        print(max_iter_per_epoch)
        pre_loss=None
        self.project_mat = self.project_mat.detach()
        self.corr_mask = self.corr_mask.detach()
        for n, (image_gt,uvz_gt, project_info) in enumerate(self.train_loader):
            
            b,c,h,w=image_gt.size()

            #if b!=self.args.batch_size:
            #    continue
            #print(n,image_gt.size())
            #continue 
            #print(resize_ratio.size())
            self.total_step = self.total_step + 1
            project_mat_gt = self.project_mat 
            
            image_gt = image_gt.cuda(non_blocking=True)
            if self.args.do_sat:
                image_gt = torch.autograd.Variable(image_gt,requires_grad=True)
            project_info = {k:v.cuda(non_blocking=True) for k,v in project_info.items()}
            uvz_gt = uvz_gt.cuda(non_blocking=True)
            #project_mat_gt = project_mat_gt.cuda(non_blocking=True)
            
            self.summary_loss={}
            self.summary_image={}
            self.summary_vertices={}
            self.summary_histogram={}
           
            tensor_preds = self.model.forward(image_gt)
  
            self.uvz_preds, self.score_preds=None,None
            if self.dict_info['decoder']['do_uvz']:
                self.uvz_preds = [self.mask_uvz(tensor[0],self.corr_mask) for tensor in tensor_preds[:4]]
                uvz_gt         = self.mask_uvz(uvz_gt,self.corr_mask)
            if self.dict_info['decoder']['do_score']:
                self.score_preds = [self.mask_uvz(tensor[1],self.corr_mask) for tensor in tensor_preds[:4]]
            
            self.run_loss(self.uvz_preds,uvz_gt, image_gt, project_info, project_mat_gt,'Train')
            
           
            if self.args.do_sat:
                self.summary_loss={}
                self.summary_image={}
                self.summary_vertices={}
                self.summary_histogram={}
       
                self.optim.zero_grad()
                self.total_loss.backward()
 
                image_gt = torch.autograd.Variable(image_gt + 0.01*torch.tanh(image_gt.grad), requires_grad=False)
                tensor_preds = self.model.forward(image_gt)
                self.uvz_preds, self.score_preds=None,None
                if self.dict_info['decoder']['do_uvz']:
                    self.uvz_preds = [self.mask_uvz(tensor[0],self.corr_mask) for tensor in tensor_preds[:4]]
                    uvz_gt         = self.mask_uvz(uvz_gt,self.corr_mask)
                if self.dict_info['decoder']['do_score']:
                    self.score_preds = [self.mask_uvz(tensor[1],self.corr_mask) for tensor in tensor_preds[:4]]

                self.run_loss(self.uvz_preds,uvz_gt, image_gt, project_info, project_mat_gt,'Train')

            #ls=self.total_loss.item()
            #if ls>0.1 or pre_loss and ls > 3 * pre_loss:
            #    continue
            #pre_loss=ls



            self.syth_losses.update(self.total_loss, image_gt.shape[0])
            self.summary_loss['Train/avg_loss'] = self.syth_losses.avg
            
            self.optim.zero_grad()
            self.total_loss.backward()
            self.optim.step()
            if n % 50 == 0:
                self.model.eval()
                self.run_val_batch()
                self.model.train()
                self.write_summary()
                print('epoch:{},iter:{},total_loss:{}'.format(epoch,self.total_step,self.total_loss.cpu()))
                
            if n > max_iter_per_epoch:
                break


    def valid_epoch(self):
        self.begin_valid()           

    def begin_train(self):
        self.model.train()
  
    
    def begin_valid(self):
        self.model.eval()
     
    
    def setup_models(self):
        self.project_mat = torch.matmul(self.camera_intern, self.camera_extern)
        self.dict_info={}
        self.dict_info['encoder'] = {'num_in_layers':3,'encoder_name':args.encoder_name,'pretrained':args.encoder_pretrained}
        self.dict_info['decoder'] = {'do_uvz':args.do_uvz,'do_score':args.do_score,'camera_intern_extern':self.project_mat[:args.batch_size//len(device_ids)].clone(),'do_min_max_v':False}
        #self.dict_info['corr_mask'] = self.corr_mask
        #self.dict_info['v_to_vt'] = self.v_to_vt
        print('do_score:', self.dict_info['decoder']['do_score'])
        self.model = Model(self.dict_info)
        if self.args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
      
    def save_model(self, epoch):
        state = {
                 'state':self.model.state_dict(),
                 'optimizer':self.optim.state_dict()
                 }
        model_dir = self.args.model_dir
        prefix = 'score_' if self.args.do_score else ''
        torch.save(state, f'{model_dir}/{prefix}hand_recon_parameters_epoch{epoch}.pth')
        

    def ex_model(self,k,ex=['corr_mask','v_to_vt','camera_intern_extern','project_mat']):
        for e in ex:
            if e in k:
                return False
        return True

    def load_model(self):
        #return
        #checkpoint = torch.load(self.train_cfg.TRAIN.LOAD_PATH)
        model_path = self.load_path
        #'''
        #if self.dict_info['decoder']['do_score']:
        #    model_path = model_path[:-4]+'_score.pth'
        pretrained_dict_ = torch.load(model_path)['state']
        #import pdb;pdb.set_trace()
        model_dict = self.model.state_dict()
        #import pdb;pdb.set_trace()
        predict='module.' if self.args.use_multiple_gpu else ''

        
        pretrained_dict = {(k if 'module' in k else predict+k):v for k, v in pretrained_dict_.items() if self.ex_model(k)}
        print('pretrain nodes:',len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        #else:
        #'''
        #import pdb;pdb.set_trace()
        #checkpoint = torch.load(model_path)
        #print('len state is :', len(checkpoint['state']))
        #self.model.load_state_dict(checkpoint['state'])
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
