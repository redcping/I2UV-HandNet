# -*- coding:utf-8 -*-
import torch 
import numpy as np

def sample_uv_xyz(v_to_vt, images):
    """
    v_to_vt: B x N x 4 x 2
    images： B x C x H x W
    outputs：B x N x C
    """
    sampled = torch.nn.functional.grid_sample(images, v_to_vt)
    #sampled = torch.median(sampled, dim=-1, keepdim=False)[0].permute(0, 2, 1)
    sampled = torch.mean(sampled, dim=-1, keepdim=False).permute(0, 2, 1) 
    return sampled

def sample_uv_xyz1(v_to_vt, images):
    """
    v_to_vt: B x N x 4 x 2
    images： B x C x H x W
    outputs：B x N x C
    """
    sampled = torch.nn.functional.grid_sample(images, v_to_vt)
    max_v = torch.max(torch.abs(sampled),dim=-1,keepdim=True)[0].repeat(1,1,1,sampled.size()[-1])
    masked =  torch.where(max_v == torch.abs(sampled), sampled, - max_v-1.)
    sampled = torch.max(masked, dim=-1, keepdim=False)[0].permute(0, 2, 1) 
    return sampled



def compute_theta1(vertices, vertices_gt):
    v    = vertices - torch.mean(vertices,dim=1,keepdim=True)
    v_gt = vertices_gt - torch.mean(vertices_gt,dim=1,keepdim=True)
    vt = v.permute(0, 2, 1)
    vtv = torch.matmul(vt, v)
    vtv_inv = torch.inverse(vtv)
    vtv_inv_vt = torch.matmul(vtv_inv, vt)
    theta = torch.matmul(vtv_inv_vt, v_gt)
    return theta




def compute_theta(xy1, xy2, mask):
    '''
    xy1,xy2: [x1,y1],[x2,y2] B,1,H,W
    mask: B, 1, H, W
    return：B,3,3
    '''
    x1, y1 = xy1
    x2, y2 = xy2
    x1x2, y1y2 = x1*x2, y1*y2
    x2y1, x1y1 = x2*y1, x1*y1
    x1x1, y1y1 = x1*x1, y1*y1

    a = (x1x2 + y1y2)/(x1x1 + y1y1 + 1e-6) * mask
    b = (y1y2 - x2y1)/(x1y1 + y1y1 + 1e-6) * mask
    
    
    a = torch.sum(torch.clamp(a, -1, 1), dim=(2,3))/torch.sum(mask.view(mask.size()[0],1,-1),dim=-1) #b 1
    b = torch.sum(torch.clamp(b, -1, 1), dim=(2,3))/torch.sum(mask.view(mask.size()[0],1,-1),dim=-1) #b 1

    return a, b
    
    
def compute_R_and_theta(prediction, target, mask):
    '''
    prediction,target: B, 3, H, W
    mask: B, 1, H, W
    return：B,3,3
    '''
    
    b, c, h, w = prediction.size()
    prediction = prediction - (torch.sum(prediction.view(b,c,-1),dim=-1)/torch.sum(mask.view(b,1,-1),dim=-1)).view(b,c,1,1)
    target     = target - (torch.sum(target.view(b,c,-1),dim=-1)/torch.sum(mask.view(b,1,-1),dim=-1)).view(b,c,1,1)
    
    x1, y1, z1 = torch.split(prediction*mask,[1,1,1], dim=1)
    x2, y2, z2 = torch.split(target*mask,    [1,1,1], dim=1)
    
    rza, rzb =  compute_theta([x1,y1],[x2,y2], mask)  # [[a,-b],[b,a]]
    rya, ryb =  compute_theta([z1,x1],[z2,x2], mask)  # [[a,b],[-b,a]]
    rxa, rxb =  compute_theta([y1,z1],[y2,z2], mask)  # [[a,-b],[b,a]]
    ones, zeros = torch.ones_like(rza).cuda(), torch.zeros_like(rza).cuda()
    
    Rx=[ones, zeros, zeros, zeros, rxa,  -rxb,  zeros, -rxb,   rxa]
    Ry=[rya,  zeros, ryb,   zeros, ones, zeros, -ryb,  zeros,  rya]
    Rz=[rza,  -rzb,  zeros, rzb,   rza,  zeros, zeros, zeros, ones]

    Rx = torch.stack(Rx,dim=-1).view(b,3,3) 
    Ry = torch.stack(Ry,dim=-1).view(b,3,3) 
    Rz = torch.stack(Rz,dim=-1).view(b,3,3)
    
    R = torch.matmul(torch.matmul(Rx,Ry),Rz)
    
    theta_loss = [rza*rza + rzb*rzb - 1., rya*rya + ryb*ryb - 1.,rxa*rxa + rxb*rxb - 1.]

    return [R, theta_loss]


def vertices_project_map(vertices, project_mat):
    '''
    function: word_xyz(vertices)
    vertices : B,N,3
    project_mat: B,3,4 
    return uv: B,N,2    
    '''
    B, N, C = vertices.size()
    word_xyz = torch.cat((vertices, torch.ones_like(vertices[...,-1:]).cuda()),dim=1)
    word_xyz = word_xyz.reshape(-1, C + 1, 1) #B*N C+1, 1
    project_mat = project_mat.repeat(N, 1, 1)
    uvs = torch.matmul(project_mat, word_xyz).reshape(B, N, C)
    u, v, s = uvs[...,0], uvs[...,1], uvs[...,2]

    x_shifts = u / (s + 1e-8)
    y_shifts = v / (s + 1e-8)

    return torch.stack((x_shifts, y_shifts), dim=-1)

def project_map(uvz, project_mat, r = 1):
    '''
    function: word_xyz(uvz) to uvs
    word_xyz : B,C,H,W
    project_mat: B,3,4 
    return uv: B,2,H,W 
    '''
    B, C, H, W = uvz.size()
    word_xyz = torch.cat((uvz, torch.ones_like(uvz[:,-1:,:,:]).cuda()),dim=1)
    word_xyz = word_xyz.permute(0,2,3,1).reshape(-1, C + 1, 1) #B*H*W C 1
    project_mat = project_mat.repeat(H * W, 1, 1)
    uvs = torch.matmul(project_mat, word_xyz).reshape(B, H, W, C)
    u, v, s = uvs[...,0], uvs[...,1], uvs[...,2]

    x_shifts = u / (s * W * r + 1e-8)
    y_shifts = v / (s * H * r + 1e-8)

    return torch.stack((x_shifts, y_shifts), dim=3).permute(0,3,1,2)

def get_top_left(vertices_uv, crop_r=1.2):
    '''
    vertices_uv : B N 2
    return top_left: B 2
    '''
    max_uv, min_uv = torch.max(vertices_uv,dim=1)[0], torch.min(vertices_uv,dim=1)[0]
    center = (max_uv + min_uv)/2
    size   = torch.max(max_uv - min_uv, dim=-1,keepdim=True)[0]
    top_left = center - size * crop_r/2
    return top_left

def project_map1(uvz, project_mat, r = 1,v_to_vt=None):
    uv = project_map(uvz, project_mat, r)
    return uv
    vertices_uv = sample_uv_xyz(v_to_vt, uv) #B N 2
    uv = uv - get_top_left(vertices_uv,1.).view(-1,2,1,1)
    return uv

def get_project_map1(uvz, project_mat, v_to_vt, resize_ratio, r):
    '''
    uvz: B 3 H W
    project_mat: B 3 4
    v_to_vt: B M 4
    resize_ratio: B 1
    r: float
    return uv: B 2 H W
    '''
    uv = project_map(uvz, project_mat, r)
    vertices_uv = sample_uv_xyz(v_to_vt, uv) #B N 2
    uv = uv - get_top_left(vertices_uv).view(-1,2,1,1)
    uv =  uv * resize_ratio.view(-1,1,1,1)
    return uv


def get_project_map(uvz, project_mat, v_to_vt, r, project_info):
    '''
    uvz: B 3 H W
    project_mat: B 3 4
    v_to_vt: B M 4
    resize_ratio: B 1
    r: float
    return uv: B 2 H W
    '''
    uv = project_map(uvz, project_mat, r)
    vertices_uv = sample_uv_xyz(v_to_vt, uv) #B N 2
    uv = uv - get_top_left(vertices_uv,project_info['crop_r'].view(-1,1)).view(-1,2,1,1)
    uv =  uv * project_info['resize_ratio'].view(-1,1,1,1) + project_info['duv'].view(-1,2,1,1)
    uv = torch.clamp(uv,0,1)
    return uv



def normalize_vertices(vertices):
    vertices -= vertices.min(1, keepdim=True)[0]
    vertices /= torch.abs(vertices).max(1)[0].max(1)[0][:,None,None]
    vertices *= 2
    vertices -= vertices.max(1, keepdim=True)[0]/ 2
    return vertices

def compute_scale_and_shift(prediction, target, mask):
    scale = (1.0 * prediction.size()[-1])/target.size()[-1]
    target = torch.nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
    mask = torch.nn.functional.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=True)
    a_00 = torch.sum(mask * prediction * prediction, dim=(2, 3),keepdim=True)
    a_01 = torch.sum(mask * prediction, dim=(2, 3),keepdim=True)
    a_11 = torch.sum(mask, dim=(2, 3), keepdim=True)

    b_0 = torch.sum(mask * prediction * target, dim=(2, 3),keepdim=True)
    b_1 = torch.sum(mask * target, dim=(2, 3),keepdim=True)

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero().detach()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def v_to_vt_align(v_to_vt, texcoords):
    v_to_vt = list(v_to_vt)
    c3dto2d=[]
    max_len=-1
    for li in v_to_vt:
        li = list(li)
        x = len(li)
        max_len=max(x,max_len)
        ans = li
        if x==1:
            ans = [li[0],li[0],li[0],li[0]]
        elif x==2:
            ans = [li[0],li[1],li[0],li[1]]
        elif x==3:
            ans = [li[0],li[1],li[2],li[0]]
        xy = []
        for v in ans:
            xy.append(texcoords[v])
        c3dto2d.append(xy)
    result =  np.array(c3dto2d).reshape(len(c3dto2d),4,2)
    return result


def _parse_obj(obj_file='HandRightUV.obj'):
        with open(obj_file, 'r') as fin:
            lines = [l 
                for l in fin.readlines()
                if len(l.split()) > 0
                and not l.startswith('#')
            ]
        
        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []
        
        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                texcoords.append((1 - v, u))
                
        # Stack these into an array
        vertices = np.vstack(vertices).astype(np.float32)
        texcoords = np.vstack(texcoords).astype(np.float32)
        
        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        #
        # Store the texcoord faces and a mapping from texcoord faces
        # to vertex faces
        vt_faces = []
        vt_to_v = {}
        v_to_vt = [None] * vertices.shape[0]
        for i in range(vertices.shape[0]):
            v_to_vt[i] = set()

        faces = []

        for line in lines:
            vs = line.split()
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                faces.append([v0,v1,v2])
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                vt_faces.append((vt0, vt1, vt2))
                vt_to_v[vt0] = v0
                vt_to_v[vt1] = v1
                vt_to_v[vt2] = v2
                v_to_vt[v0].add(vt0)
                v_to_vt[v1].add(vt1)
                v_to_vt[v2].add(vt2)

        vt_faces = np.vstack(vt_faces)

        v_to_vt = v_to_vt_align(v_to_vt, texcoords)     
        #return 
        
        faces=np.array(faces).reshape(-1,3)
        print(v_to_vt.shape, faces.shape)

        '''
        tmp_dict = {
            'texcoords': texcoords,
            'vt_faces': vt_faces,
            'vt_to_v': vt_to_v,
            'v_to_vt': v_to_vt
        }
        return tmp_dict
        '''
        return  v_to_vt.astype(np.float32), faces.astype(np.float32)


v_to_vt,_ = _parse_obj()
np.save('./v_to_vt_new.npy',v_to_vt)

