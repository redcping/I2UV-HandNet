# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter
import config 
import metrics
import numpy as np
import torch.nn.functional as F
import utils
import argparse
from multiprocessing import Manager
import torch.nn as nn
import torch.utils.data as data
import glob
import pickle
from models import GridMask
import cv2
parser = argparse.ArgumentParser()
# for jarvis
parser.add_argument('--data_dir',                type=str,  help='Directory for storing input data')
parser.add_argument('--output_dir',              type=str,  help='Directory for storing output data')
parser.add_argument('--model_dir',               type=str,  help='Directory for storing model')
parser.add_argument('--previous_job_output_dir', type=str,  help='Directory for previous_job_output_dir', default='')
#for user
parser.add_argument('--encoder_name',       type=str,  help="encode net",                    default='resnet50')
parser.add_argument('--datanames',          type=str,  help='datanames',                     default='uvz_778_3k_pair')
parser.add_argument('--load_path',          type=str,  help='load path',                     default='hand.pth')
parser.add_argument('--init_lr',            type=float,help='init_lr',                       default = 0.0001)
parser.add_argument('--img_size',           type=int,  help="image size",                    default=256)
parser.add_argument('--batch_size',         type=int,  help="batch size",                    default=256)
parser.add_argument('--epoch',              type=int,  help="num epoch",                     default=1000)
parser.add_argument('--num_workers',        type=int,  help="num workers",                   default=10)
parser.add_argument('--encoder_pretrained', type=bool, help="encode net encoder_pretrained", default=False)
parser.add_argument('--use_multiple_gpu',   type=bool, help="use_multiple_gpu",              default=False)
parser.add_argument('--do_sat',             type=bool, help="do sat",                        default=False)
args = parser.parse_args()
device_ids=[0,1,2,3] if args.use_multiple_gpu else [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in device_ids])




def rotation_data(uvz,uvz_3k):
    i = np.random.randint(0,360)
    w = uvz.shape[0]
    M = cv2.getRotationMatrix2D((w//2,w//2),i,1)
    uvz[...,:-1] = np.matmul(np.concatenate((uvz[...,:-1]*w, np.ones_like(uvz[...,-1:])),axis=-1),M.T)/w
    uvz_3k[...,:-1] = np.matmul(np.concatenate((uvz_3k[...,:-1]*w, np.ones_like(uvz_3k[...,-1:])),axis=-1),M.T)/w
    return uvz, uvz_3k

class ToTensorTransform(object):
    def __init__(self):
        super().__init__()
    def __call__(self, pic:np.ndarray):
        if pic.ndim == 2:
            pic = pic[:,:,None]
        return torch.tensor(pic.transpose(2,0,1),dtype=torch.float32)

class HandData(data.Dataset):
    
    def __init__(self, datanames,
                 mode ='train',
                 uvz_transform=ToTensorTransform()):
        super().__init__()

        self.mode=mode 
        self.datasets = glob.glob(f'{datanames}/*.npz', recursive=True)
        
        print(len(self.datasets))
                
        manager = Manager()
        self.datasets =  manager.list(self.datasets)
        self._uvz_transform  = uvz_transform
    
    def __getitem__(self, index):
        sample = self.datasets[index]
        info = np.load(sample)
        uvz_778, uvz_3k, max_min = info['uvz_778'],info['uvz_3k'],info['max_min']
        
        uvz_778, uvz_3k = rotation_data(uvz_778, uvz_3k)
        if self._uvz_transform is not None:
            uvz_778 = self._uvz_transform(uvz_778)
            uvz_3k = self._uvz_transform(uvz_3k)
        
        max_min_v = torch.tensor(np.array(max_min,dtype=np.float32).reshape(2,3),dtype=torch.float32)
        
        return uvz_778, uvz_3k, max_min_v


    def __len__(self):
        return len(self.datasets)


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

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
        data_dir = data_root + '/hand/' + args.datanames
        train_data = HandData(datanames=f'{data_dir}',
                                                 mode ='train',
                                                 uvz_transform=ToTensorTransform())
       
        
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, 
                                                             shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
        
        self.size = self.args.img_size 
        #init model
        v_to_vt,faces = utils._parse_obj('./HandRightUV_3K.obj')
        v_to_vt = (v_to_vt * (self.size-1)+0.5).astype(np.int32)
        v_to_vt = v_to_vt.astype(np.float32)/(self.size-1)
        self.v_to_vt = torch.from_numpy(v_to_vt.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()
        self.v_to_vt = 2 * torch.cat((self.v_to_vt[...,1:2], self.v_to_vt[...,0:1]),-1) - 1
        self.GridMask = GridMask()
        self.faces = torch.from_numpy(faces.astype(np.float32)[None,...].repeat(self.args.batch_size, axis=0)).cuda()

        corr_mask = np.sum(np.asarray(Image.open('./HandRightUV_Corr.png').convert('RGB'), dtype=np.float32),axis=-1,keepdims=True) > 0
        corr_mask = corr_mask.astype(np.float32)[None,...].transpose(0,3,1,2)
        self.corr_mask = torch.from_numpy(corr_mask.repeat(self.args.batch_size, axis=0)).cuda().float()
        self.corr_mask = torch.nn.functional.interpolate(self.corr_mask, scale_factor=self.size/256, mode='bilinear', align_corners=True)
        self.setup_models()
        self.lr = self.args.init_lr
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.load_pretrain()
        self.total_step=-1
        self.max_iter_step = len(self.train_loader)* self.args.epoch
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

   
    def mask_uvz(self, uvz, mask):
        scale = (1.0 * uvz.size()[-1])/mask.size()[-1]
        mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
        return uvz * mask
   

    def run_loss(self, uvz_pred, uvz_778, uvz_3k, max_min_v, mode='Train'):
    
        b, c, h, w    = uvz_pred.size()
        #vertices_778  = utils.sample_uv_xyz(self.v_to_vt, uvz_778)
        vertices_3k   = utils.sample_uv_xyz(self.v_to_vt, uvz_3k)
        vertices_778  = vertices_3k[:,:778]
        vertices_pred = utils.sample_uv_xyz(self.v_to_vt, uvz_pred)
      
        
        #losses


        '''uvz_loss'''
        self.uvz_loss = self.l1_loss_fn(uvz_pred, uvz_3k)

        '''points loss'''
        self.point_loss = self.point_loss_fn(vertices_pred[:,:778], vertices_778[:,:778]) + 2 * self.point_loss_fn(vertices_pred[:,778:], vertices_3k[:,778:])
        

        '''total_loss'''
        self.total_loss = self.uvz_loss +  0.01 * self.point_loss 

        '''summary loss'''
        self.summary_loss['{}/uvz_loss'.format(mode)]      = self.uvz_loss
        self.summary_loss['{}/point_loss'.format(mode)]    = self.point_loss 
        self.summary_loss['{}/total_loss'.format(mode)]    = self.total_loss

        '''summary image'''
        self.summary_image['{}/uvz_pred'.format(mode)]     = uvz_pred
        self.summary_image['{}/uvz_778'.format(mode)]      = uvz_778
        self.summary_image['{}/uvz_3k'.format(mode)]       = uvz_3k
        self.summary_image['{}/d_uvz'.format(mode)]        = torch.abs(torch.mean(uvz_3k - uvz_pred,dim=1,keepdim=True))

        '''summary vertices'''
        self.summary_vertices['{}/vertices_pred'.format(mode)]  = vertices_pred
        self.summary_vertices['{}/vertices_3k'.format(mode)]    = vertices_3k

        '''summary histogram'''
        #self.summary_histogram['{}/uvz'.format(mode)]    = uvz_pred


    def write_summary(self):
        k=1
        self.summary_writer.add_scalar('Train/lr', self.lr, self.total_step)
        if True:
            for key, value in self.summary_vertices.items():
                self.summary_writer.add_mesh('{}'.format(key), utils.normalize_vertices(value[:k]), colors=255*torch.ones(value[:k].shape, dtype=torch.uint8),  faces=self.faces[:k], global_step=self.total_step)

            for key, value in self.summary_image.items():
                self.summary_writer.add_images('{}'.format(key), value[:k], self.total_step)

            for key, value in self.summary_loss.items():
                self.summary_writer.add_scalar('{}'.format(key), value.item(), self.total_step)

            for key, value in self.summary_histogram.items():
                self.summary_writer.add_histogram('{}'.format(key), value[:k], self.total_step)
        #except:
        #    pass
 
 
    def train(self):
        pre_epoch = 0
        for epoch in range(pre_epoch,self.args.epoch):
            self.GridMask.set_prob(epoch+self.args.epoch//2,self.args.epoch)
            #self.adjust_learning_rate(self.optim, epoch)
            self.train_epoch(epoch)
            if epoch % 50 == 0:
                self.save_model(epoch)

    def train_epoch(self, epoch):
        self.begin_train()
        pre_loss=None
        for n, (uvz_778, uvz_3k, max_min_v) in enumerate(self.train_loader):
            
            b,c,h,w=uvz_778.size()
            self.total_step = self.total_step + 1
            if self.lr>1e-6:
                self.adjust_learning_rate(self.optim)
            #if b!=self.args.batch_size:
            #    continue
            #print(n,image_gt.size())
            #continue 
            #print(resize_ratio.size())
         
          
            uvz_778 = uvz_778.cuda(non_blocking=True)*self.corr_mask
            if self.args.do_sat:
                uvz_778 = torch.autograd.Variable(uvz_778,requires_grad=True)
            uvz_3k = uvz_3k.cuda(non_blocking=True)*self.corr_mask
            max_min_v = max_min_v.cuda(non_blocking=True)
            #uvz_778 = self.GridMask(uvz_778) 
            self.summary_loss={}
            self.summary_image={}
            self.summary_vertices={}
            self.summary_histogram={}
           
            uvz_pred = self.model.forward(uvz_778) * self.corr_mask
                
            self.run_loss(uvz_pred, uvz_778, uvz_3k, max_min_v,'Train')




            self.syth_losses.update(self.total_loss, uvz_778.shape[0])
            self.summary_loss['Train/avg_loss'] = self.syth_losses.avg
            
            self.optim.zero_grad()
            self.total_loss.backward()
            self.optim.step()
            if self.total_step % 2 == 0:
                #self.model.eval()
                #self.run_val_batch()
                #self.model.train()
                self.write_summary()
                print('epoch:{},iter:{},total_loss:{}'.format(epoch,self.total_step,self.total_loss.cpu()))
                


    def valid_epoch(self):
        self.begin_valid()           

    def begin_train(self):
        self.model.train()
  
    
    def begin_valid(self):
        self.model.eval()
     
    
    def setup_models(self):
        self.model = SRCNN()
        if self.args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        
    def save_model(self, epoch):
        state = {
                 'state':self.model.state_dict(),
                 'optimizer':self.optim.state_dict()
                 }
        model_dir = self.args.model_dir
        torch.save(state, f'{model_dir}/hand_recon_super_parameters_epoch{epoch}.pth')
        

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


        pretrained_dict = {(k if 'module' in k else predict+k):v for k, v in pretrained_dict_.items()}
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

