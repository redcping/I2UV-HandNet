# -*- coding:utf-8 -*-
import os
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 4
_C.SYSTEM.NUM_WORKDERS = 4

_C.DATA = CN()
_C.DATA.TRAIN_SYTH = '/new_home/database/xvx-52000/'
_C.DATA.TRAIN_BG = '/media/algcd/ImageNet/Image'
_C.DATA.TRAIN_REAL = '/new_home/database/hand/training'

_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.EPOCH = 1000
_C.TRAIN.REAL_EPOCH = 100
_C.TRAIN.REAL_FREQ = 1
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.IMAGE_SIZE = 256
_C.TRAIN.SUMMARY_PATH = '/tmp/log'
_C.TRAIN.LOG_FREQ = 10
_C.TRAIN.SAVE_GAP = 1
_C.TRAIN.SAVE_PATH = '/tmp/log'
_C.TRAIN.LOAD_PATH = '/home/algcd/Dev/HandMapping/meta/mbv3_large_old.pth'

_C.LOSSES = CN()
_C.LOSSES.LAMBDA_CHAMFIER = 0.5
_C.LOSSES.LAMBDA_NORM = 0.5
_C.LOSSES.LAMBDA_TEX_RENDER = 0.5
_C.LOSSES.LAMBDA_CORR_RENDER = 0.5
_C.LOSSES.LAMBDA_DEPTH = 0.5
_C.LOSSES.LAMBDA_GEOMTRY = 0.5
_C.LOSSES.LAMBDA_GAN = 0.0
_C.LOSSES.LAMBDA_SEGM = 0.5
_C.LOSSES.LAMBDA_DT = 0.5
_C.LOSSES.LAMBDA_UV = 0.5

_C.AUX = CN()
_C.AUX.OBJ_PATH = '/home/algcd/Dev/HandMapping/meta/HandRightUV.obj'
_C.AUX.UVS_PATH = '/home/algcd/Dev/HandMapping/meta/HandRightUV_UV.png'
_C.AUX.UVS_MASK_PATH = '/home/algcd/Dev/HandMapping/meta/HandRightUV_UV_mask.png'
_C.AUX.MANO_PATH = '/home/algcd/Dev/HandMapping/meta/mano'
_C.AUX.CAMERA_MATRIX = '/new_home/database/xvx/camera_info.npz'
_C.AUX.MANO_POSE_NCOMPS = 6
_C.AUX.MANO_SHAPE_NCOMPS = 10
_C.AUX.MANO_SIDE = 'right'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def load_cfg_file(yaml_path):
    cfg = get_cfg_defaults()
    assert os.path.exists(yaml_path), 'config file exists'
    cfg.merge_from_file(yaml_path)
    cfg.freeze()
    return cfg

    

