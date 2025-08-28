# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def get_xy_gradients(img):
    gradients_x = img[:,:,:,:-1] - img[:,:,:,1:]
    gradients_y = img[:,:,:-1,:] - img[:,:,1:,:]
    return gradients_x,gradients_y

def smoothness_loss(img_pred, img, score):

    scale = (1.0 * img_pred.size()[-1])/img.size()[-1]
    img = nn.functional.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=True)
    score = nn.functional.interpolate(score, scale_factor=scale, mode='bilinear', align_corners=True)
    gradients_x, gradients_y = get_xy_gradients(img_pred)
    gt_gradients_x, gt_gradients_y = get_xy_gradients(img)
    smoothness_x = torch.mean(torch.abs(gt_gradients_x - gradients_x)*score[:,:,:,:-1])
    smoothness_y = torch.mean(torch.abs(gt_gradients_y - gradients_y)*score[:,:,:-1,:])
    smoothness = smoothness_x + smoothness_y
    return smoothness

def reg_loss(prediction, target):
    scale = (1.0 * prediction.size()[-1])/target.size()[-1]
    target = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
    gx, gy = get_xy_gradients(prediction - target)
    return torch.mean(torch.abs(gx)) + torch.mean(torch.abs(gy))

def mse_loss(prediction, target):
    scale = (1.0 * prediction.size()[-1])/target.size()[-1]
    target = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
    loss = torch.nn.MSELoss()
    return loss(prediction,target)

def l1_loss(prediction, target):
    scale = (1.0 * prediction.size()[-1])/target.size()[-1]
    target = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
    loss = torch.nn.L1Loss()
    return loss(prediction,target)


def vertex_loss(mesh_pred, mesh_gt):
    loss = nn.functional.mse_loss(mesh_pred['vertices'], mesh_gt['vertices'])
    loss += nn.functional.mse_loss(mesh_pred['joints'], mesh_gt['joints'])*2.0
    loss += torch.norm(mesh_pred['pose'], dim=1).mean()*10.0+\
            torch.norm(mesh_pred['shape'], dim=1).mean()*20.0
    #loss += kal.metrics.mesh.edge_length(vertex_pred)
    #loss += kal.metrics.mesh.laplacian_loss(vertex_pred, vertex_gt)
    return loss
        

def dice_coefficient_loss(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss



def laplace_loss(prediction, target, logb):    
    assert logb.dim()==4
    scale = (1.0 * prediction.size()[-1])/target.size()[-1]
    target = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=True)
    norm = (prediction - target).norm(dim=1, keepdim=True)
    # constrain range of logb
    logb = 3.0 * torch.tanh(logb / 3.0)
    losses = 0.694 + logb + norm * torch.exp(-logb)
    return losses.mean()
