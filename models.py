from __future__ import absolute_import, division, print_function
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import utils
from manopth.manolayer import ManoLayer

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


def activate_fn(x,inplace=True):
    return F.relu(x,inplace=inplace)

def grid_sample(img, word_xyz, camera_intern_extern, r=1):
    '''
    function: word_xyz to uvs
    word_xyz : B,C,H,W
    camera_intern_extern: B,3,4        
    '''
    word_xyz = word_xyz[:,:3,:,:]
    B, C, H, W = word_xyz.size()
    #camera_intern_extern = torch.cat((camera_intern_extern[...,:-1]/r, camera_intern_extern[...,-1:]).cuda()),dim=-1)
    word_xyz = torch.cat((word_xyz, torch.ones_like(word_xyz[:,-1:,:,:]).cuda()),dim=1)
    word_xyz = word_xyz.permute(0,2,3,1).reshape(-1, C + 1, 1) #B*H*W C 1
    camera_intern_extern = camera_intern_extern.repeat(H * W, 1, 1)
    uvs = torch.matmul(camera_intern_extern, word_xyz).reshape(B, H, W, C)
    u, v, s = uvs[...,0], uvs[...,1], uvs[...,2]

    x_shifts = u / (s * W * r + 1e-8)
    y_shifts = v / (s * H * r + 1e-8)

    flow_field = torch.stack((x_shifts, y_shifts), dim=3)
    flow_field = torch.clamp(flow_field, 0., 1.)
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output


class GridMask(nn.Module):
    def __init__(self, use_h=True, use_w=True, rotate = 1, offset=False, ratio = 0.5, mode=1, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
    def set_prob(self, epoch, max_epoch):
        self.prob = 0.7#self.st_prob * epoch / max_epoch

    def forward(self, x):
        if np.random.rand() > self.prob:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, w//4)
        #d = self.d
        #self.l = int(d*self.ratio+0.5)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
#        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 
        return x.view(n,c,h,w)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return activate_fn(x)


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return activate_fn(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return activate_fn(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
 
    def forward(self, x):
        return self.func(x)

class get_tensor(nn.Module):
    def __init__(self, num_in_layers, out_channels, activate_fn=Lambda(lambda x: x)):
        super(get_tensor, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, out_channels, kernel_size=3, stride=1)
        self.activate_fn = activate_fn

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.activate_fn(x)
        return x

class LinearModel(nn.Module):
    def __init__(self, input_channels, out_channels=22, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.fc_layer=nn.Sequential(nn.Linear(input_channels, 1024),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(p_dropout),
                                             nn.Linear(1024, 512),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(p_dropout),
                                             nn.Linear(512, out_channels))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
    def forward(self, x):
        return self.fc_layer(x)

class ManoCascadeDecoder(nn.Module):    
    def __init__(self, dict_info):
        super(ManoCascadeDecoder, self).__init__()
        self.filters      = dict_info['filters']
        self.in_channels  = 128
        self.out_channels = 22#dict_info['out_channels']
        self.p_dropout    = 0.5#dict_info['p_dropout']
        self.num_joints   = 21
        self.stage        = len(self.filters)
        
        self.index, self.filters = [0,self.filters[::-1]] if dict_info['flip'] else [-1,self.filters]
        print(self.index, self.filters)
        self.convs = [nn.Conv2d(f, self.in_channels, kernel_size=1, stride=1, padding=1) for f in self.filters]
        self.convs = nn.ModuleList(self.convs)
        self.init_conv    = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels * 4, kernel_size=3, stride=2),
                                          nn.BatchNorm2d(self.in_channels*4),
                                          nn.ReLU(inplace=True))
        
        self.init_pose_shape_layer = LinearModel(self.in_channels * 4, self.out_channels, self.p_dropout)
        
        self.mano_layer   = ManoLayer(center_idx=0, side= 'right', mano_root='./mano/models/', use_pca=True, ncomps=6)
    
        self.pose_shape_layers = [LinearModel(self.num_joints * self.in_channels,self.out_channels, self.p_dropout) for i in range(self.stage)]
        self.pose_shape_layers = nn.ModuleList(self.pose_shape_layers)
       

    def norm_joints(self, joints):
        joints = torch.clamp(joints, 0., 1.)
        #b, n, c      = joints.shape
        #min_v = torch.min(joints,dim=1)[0].view(b,1,c)
        #joints = torch.clamp((joints - min_v),0.,1.)
        #min_v, max_v = torch.min(joints,dim=1)[0].view(b,1,c), torch.max(joints,dim=1)[0].view(b,1,c)
        #joints       = (joints - min_v)/(max_v - min_v + 1e-8)
        return joints
    
    def sampled_freature_for_fc(self, images, joints):
        v_to_vt =  2 * torch.cat((joints[...,1:2], joints[...,0:1]),-1) - 1.
        v_to_vt = v_to_vt.unsqueeze(1)
        sampled = torch.nn.functional.grid_sample(images, v_to_vt)
        return sampled.reshape(images.shape[0], -1)
    
    def forward(self, skips):
        skips = skips[-self.stage:]
        if self.index==0:
            skips=skips[::-1]

        skips= [self.convs[i](skips[i]) for i in range(self.stage)]
            
        x = self.init_conv(skips[self.index])
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        pose_shape = self.init_pose_shape_layer(x.reshape(x.shape[0],-1))
        vertices_stages, joints_stages = [],[]
        vertices, joints = self.mano_layer.forward(th_pose_coeffs = pose_shape[:,:9], th_betas = pose_shape[:,9:-3], th_trans=pose_shape[:,-3:])
        vertices, joints = self.norm_joints(vertices), self.norm_joints(joints)
        vertices_stages.append(vertices)
        joints_stages.append(joints)
        for i in range(self.stage):
            fc = self.sampled_freature_for_fc(skips[i], joints)
            #print(fc.shape,i)
            pose_shape += self.pose_shape_layers[i](fc)
            vertices, joints = self.mano_layer.forward(th_pose_coeffs = pose_shape[:,:9], th_betas = pose_shape[:,9:-3], th_trans=pose_shape[:,-3:])
            vertices, joints = self.norm_joints(vertices), self.norm_joints(joints)
            vertices_stages.append(vertices)
            joints_stages.append(joints)
            
        outputs = dict(vertices_stages=vertices_stages[::-1], joints_stages=joints_stages[::-1],pose_shape_trans=pose_shape)
        return outputs


class Decoder(nn.Module):
    def __init__(self, dict_info):
        super(Decoder, self).__init__()
        self.filters = dict_info['filters']
        self.out_channels = dict_info['out_channels']
        self.do_z = dict_info['do_z'] if 'do_z' in dict_info.keys() else False
        print('do_z:',self.do_z)
        self.activate_fn = nn.Sigmoid() if self.do_z else Lambda(lambda x: x)

        # decoder
        self.upconv6 = upconv(self.filters[3], 512, 3, 2)
        self.iconv6 = conv(self.filters[2] + 512, 512, 3, 1)  #H/16
        self.out6_layer = get_tensor(512,self.out_channels, self.activate_fn) 
      
        self.upconv5 = upconv(512, 256, 3, 2)   
        self.iconv5 = conv(self.filters[1] + 256, 256, 3, 1)  #H/8
        self.out5_layer = get_tensor(256,self.out_channels, self.activate_fn)

        self.upconv4 = upconv(256, 128, 3, 2)   
        self.iconv4 = conv(self.filters[0]+128, 128, 3, 1)
        self.out4_layer = get_tensor(128, self.out_channels, self.activate_fn)       #H/4

        self.upconv3 = upconv(128, 64, 3, 1)
        self.iconv3 = conv(64 + 64 + self.out_channels, 64, 3, 1)
        self.out3_layer = get_tensor(64, self.out_channels, self.activate_fn)      #H/4

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + self.out_channels, 32, 3, 1)
        self.out2_layer = get_tensor(32,self.out_channels, self.activate_fn)      #H/2

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + self.out_channels, 16, 3, 1)
        self.out1_layer = get_tensor(16, self.out_channels, self.activate_fn)     #H

    def scale_uvz(self, uvz, r):
        return nn.functional.interpolate(uvz, scale_factor=1./r, mode='bilinear', align_corners=True)


    def forward(self, skips, camera_intern_extern,uvz_gt=None):
        # skips
        skip1 = skips[0]  #H/2
        skip2 = skips[1]  #H/4
        skip3 = skips[2]  #H/4
        skip4 = skips[3]  #H/8
        skip5 = skips[4]  #H/16
        x5    = skips[5]  #H/32
        self.camera_intern_extern = camera_intern_extern
        stage = 2
        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)
        self.out6 = self.out6_layer(iconv6) #H/16
        self.uout6 = nn.functional.interpolate(self.out6, scale_factor=2, mode='bilinear', align_corners=True) #H/8 wa
        
         
         
        uout6 = self.scale_uvz(uvz_gt, 8) if uvz_gt is not None and stage>5 else self.uout6
        #uout6 = self.uout6
        #upconv5 =  self.upconv5(iconv6)
        upconv5 = grid_sample(self.upconv5(iconv6), uout6, self.camera_intern_extern, 8)
        skip4_   =  grid_sample(skip4, uout6, self.camera_intern_extern, 8)
        
        concat5 = torch.cat((upconv5, skip4_), 1)
        iconv5 = self.iconv5(concat5)
        self.out5 = self.out5_layer(iconv5) #H/8
        self.uout5 = nn.functional.interpolate(self.out5, scale_factor=2, mode='bilinear', align_corners=True)  #H/4 ok
    

        uout5 = self.scale_uvz(uvz_gt, 4) if uvz_gt is not None and stage>4 else self.uout5
        #uout5 = self.uout5
        upconv4 = self.upconv4(iconv5)
        skip3_   =  grid_sample(skip3, uout5, self.camera_intern_extern, 4)
        concat4 = torch.cat((upconv4, skip3_), 1)
        iconv4 = self.iconv4(concat4)
        self.out4 = self.out4_layer(iconv4) #H/4
        self.uout4 = nn.functional.interpolate(self.out4, scale_factor=1, mode='bilinear', align_corners=True) #H/4 ok
        self.out4 = nn.functional.interpolate(self.out4, scale_factor=0.5, mode='bilinear', align_corners=True) #H/8
    

        uout4 = self.scale_uvz(uvz_gt, 4) if uvz_gt is not None and stage > 3 else self.uout4
        #uout4 = self.uout4
        #for i in range(2):          
        upconv3 = self.upconv3(iconv4)
        skip2_  =  grid_sample(skip2, uout4, self.camera_intern_extern, 4)
        #print(self.uout4.shape,skip2_.shape)
        concat3 = torch.cat((upconv3, skip2_, self.uout4), 1)
        iconv3 = self.iconv3(concat3)
        self.out3 = self.out3_layer(iconv3) #H/4
        self.uout3 = nn.functional.interpolate(self.out3, scale_factor=2, mode='bilinear', align_corners=True) #H/2 wa
    
        uout3 = self.scale_uvz(uvz_gt, 2) if uvz_gt is not None and stage > 2 else self.uout3
        #uout3 = self.uout3

        upconv2 = self.upconv2(iconv3)
        skip1_  =  grid_sample(skip1, uout3, self.camera_intern_extern, 2)
        concat2 = torch.cat((upconv2, skip1_, self.uout3), 1)
        iconv2 = self.iconv2(concat2)
        self.out2 = self.out2_layer(iconv2) #H/2
        self.uout2 = nn.functional.interpolate(self.out2, scale_factor=2, mode='bilinear', align_corners=True) #H ok
    
        uout2 = self.scale_uvz(uvz_gt, 1) if uvz_gt is not None else self.uout2
        #uout2 = self.uout2
        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, uout2), 1)
        iconv1 = self.iconv1(concat1)
        self.out1 = self.out1_layer(iconv1) #H
        #self.out2 = self.out3
        #self.out1 = self.out2
        return [self.out1, self.out2, self.out3,self.out4, self.out5, self.out6]




class ResnetEncoder(nn.Module):
    def __init__(self, num_in_layers, encoder='resnet18', pretrained=False):
        super(ResnetEncoder, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50',\
                           'resnet101', 'resnet152'],\
                           "Incorrect encoder type"
        resnet = class_for_name("torchvision.models", encoder)\
                                (pretrained=False)
        if pretrained:
            model_dict = torch.load('./{}.pth'.format(encoder))
            print(encoder,len(model_dict.keys()))
            resnet.load_state_dict(model_dict)
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                              kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1 # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool # H/4

        # encoder
        self.encoder1 = resnet.layer1 # H/4
        self.encoder2 = resnet.layer2 # H/8
        self.encoder3 = resnet.layer3 # H/16
        self.encoder4 = resnet.layer4 # H/32
        #'''
        cnt=0
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                cnt+=1
                layer.eval()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
        print('total_bn:{}'.format(cnt))
        #'''

    def forward(self, x):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return [x_first_conv, x_pool1, x1, x2, x3, x4]

class Encoder(nn.Module):
    def __init__(self, dict_info):
        super(Encoder, self).__init__()
        self.num_in_layers = dict_info['num_in_layers']
        self.encoder_name  = dict_info['encoder_name']
        self.pretrained    = dict_info['pretrained']
        if 'resnet' in self.encoder_name:
            print(self.encoder_name)
            self.encoder_layer = ResnetEncoder(self.num_in_layers , self.encoder_name, self.pretrained)
        else:
            self.encoder_layer = None

    def forward(self, x):
        return self.encoder_layer(x)







class ManoModel(nn.Module):    
    def __init__(self, mano_fc_in,side='right',mano_path='./mano/models/'):
        super(ManoModel, self).__init__()
        
        self.mano_pose_shape = nn.Sequential(nn.Linear(mano_fc_in, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, 58))

        self.mano_render = ManoLayer(center_idx=0, side='right', mano_root=mano_path, use_pca=True, ncomps=45)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, vertices, th_trans = None):
        vertices_fc = vertices.reshape(vertices.shape[0],-1)

        th_pose_shape = self.mano_pose_shape(vertices_fc)
  
        th_pose = th_pose_shape[:,:-10]
        th_shape = th_pose_shape[:,-10:]

        vertices, joints = self.mano_render.forward(th_pose_coeffs = th_pose, th_betas = th_shape, th_trans=th_trans)
        outputs = dict(pose=th_pose, shape=th_shape, vertices=vertices, joints=joints, trans=th_trans)
        return outputs

class LSDiscriminator(nn.Module):
    def __init__(self,in_channels=3, out_channels=1, image_size = 256):
        super(LSDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 6
        self.adv_layer = nn.Linear(256 * ds_size ** 2, out_channels)


    def norm_input(self, info):
        img       = info['img']
        mask      = info['mask']
        v_to_vt   = info['v_to_vt']
        vertices  = utils.sample_uv_xyz(v_to_vt, img*mask)
        min_v, max_v = torch.min(vertices,dim=1)[0].view(-1,3,1,1),torch.max(vertices,dim=1)[0].view(-1,3,1,1)
        img = (img-min_v)/(max_v-min_v + 1e-8) * mask
        return img
        

    def forward(self, info):
        img = self.norm_input(info) 
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class Model(nn.Module):
    def __init__(self, dict_info):
        super(Model, self).__init__()
        '''
        dict_info.keys()=['encoder','decoder']
        dict_info['encoder'].keys()=['num_in_layers','encoder_name','pretrained']
        dict_info['decoder'].keys()=['do_uvz','do_score','do_z']
        '''
        self.dict_info = dict_info
        self.mano_decoder = None
        self.mano_cascade_decoder=None
        self.dis_layer = None
        
        if 'do_mano' in self.dict_info.keys() and self.dict_info['do_mano']:
            self.mano_decoder = ManoModel(778*3)
            
        if 'do_gan' in self.dict_info.keys() and self.dict_info['do_gan']:
            self.dis_layer = LSDiscriminator(in_channels=3, out_channels=1, image_size = dict_info['image_size'])

        if self.dict_info['encoder']['encoder_name'] in ['resnet18', 'resnet34']:
            self.dict_info['decoder']['filters'] = [64, 128, 256, 512]
        else:
            self.dict_info['decoder']['filters'] = [256, 512, 1024, 2048]

        self.decoder_split=[]

        if self.dict_info['decoder']['do_uvz']:
            self.decoder_split.append(3)
        if self.dict_info['decoder']['do_score']:
            self.decoder_split.append(1)
        self.dict_info['decoder']['out_channels'] = np.sum(self.decoder_split)
        # encoder  num_in_layers, encoder='resnet18', pretrained
        self.encoder = Encoder(self.dict_info['encoder'])
        # decoder out_channels=3, filters=[64,128,256,512], camera_intern_extern=None
        if 'do_mano_cascade' in self.dict_info.keys() and self.dict_info['do_mano_cascade']:
            mano_cascade_info = dict(filters=self.dict_info['decoder']['filters'],flip=self.dict_info['do_mano_cascade_flip'])
            self.mano_cascade_decoder = ManoCascadeDecoder(mano_cascade_info)
        else:
            self.decoder = Decoder(self.dict_info['decoder'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, run_info):
        # encoder
        skips = self.encoder(run_info['image'])
        
        if self.mano_cascade_decoder is not None:
            return self.mano_cascade_decoder(skips)
        real = run_info['real'] if 'real' in run_info.keys() else None
        tensor_list = self.decoder(skips, run_info['project_mat'], real)
        tensor_list = [torch.nn.functional.interpolate(t, scale_factor=256./t.size()[-1], mode='bilinear', align_corners=True) for t in tensor_list]
        outputs = dict()
        
        
            
        
        if len(self.decoder_split)>1:
            tensor_list = [torch.split(tensor, self.decoder_split, dim=1) for tensor in tensor_list]
            outputs['uvz_preds'] = [t[0] for t in tensor_list]
            outputs['score_preds'] = [t[1] for t in tensor_list]
        else:
            outputs['uvz_preds'] = tensor_list
        full_uvz = outputs['uvz_preds'][0]
        
        if 'real' in run_info.keys() and self.dis_layer is not None: 
            gan_info        = dict(img=run_info['real'], mask=run_info['corr_mask'], v_to_vt=run_info['v_to_vt'])
            real_x          = self.dis_layer(gan_info)
            gan_info['img'] = full_uvz
            gen_x           = self.dis_layer(gan_info)
            gan_info['img'] = full_uvz.detach()
            gen_x_detach    = self.dis_layer(gan_info)
            outputs['adversarial'] = dict(real_x=real_x, gen_x=gen_x, gen_x_detach=gen_x_detach)
            
        if self.mano_decoder is not None:
            vertices        = utils.sample_uv_xyz(run_info['v_to_vt'], full_uvz * run_info['corr_mask'])
            outputs['mano'] = self.mano_decoder(vertices[:,:778]) 
        return outputs


