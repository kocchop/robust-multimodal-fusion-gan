"""
Some utility functions
"""
import torch
import numpy as np
import os
from torchvision.utils import make_grid
from torchvision import transforms

import logging
import cv2
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib
import matplotlib.pyplot as plt

# stat for nyu_v2 dataset
rgb_mean = np.array([0.46797212, 0.39178501, 0.37140868])
rgb_std = np.array([0.18536709, 0.18698769, 0.19171533])

depth_mean = np.array([3.1477575302124023])
depth_std = np.array([0.8675280809402466])

sparse_mean = depth_mean
sparse_std = depth_std

# generate colormap
def generate_depth_cmap(in_tensor):
    
    in_tensor = in_tensor.squeeze(1)
    depth_tensor = in_tensor.detach().cpu().numpy()
    
    colormap = plt.get_cmap('viridis')
    out_tensor = []
    
    for img in range(depth_tensor.shape[0]):
        min_val = np.amin(depth_tensor[img])
        max_val = np.amax(depth_tensor[img])
        gray = (depth_tensor[img]-min_val)/(max_val-min_val)
        # gray = depth_tensor[img]/255.0
        gray = np.clip(gray,0,1)
        heatmap = np.round(colormap(gray) * 255).astype(np.uint8)[:,:,:3]
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        out_tensor.append(heatmap)
    
    out_tensor = np.array(out_tensor)
    out_tensor = (out_tensor).astype(np.uint8)
    
    return out_tensor

# Error Metrics
def get_loss(pred, gt):

    x_mm = pred
    y_mm = gt
    
    diff_mm = x_mm-y_mm
    rmse = torch.sqrt(torch.mean(torch.pow(diff_mm, 2.0)))
    mae = torch.mean(torch.abs(diff_mm)) 
    rel = torch.mean(torch.abs(diff_mm/y_mm))

    return rmse, rel, mae

def denormalize_sparse(tensors):
    """ Denormalizes image tensors using mean and std """
    
    tensors.mul_(sparse_std[0]).add_(sparse_mean[0])
    
    return torch.clamp(tensors, 0, 10)

def denormalize_dense(tensors):
    """ Denormalizes image tensors using mean and std """
    
    tensors.mul_(depth_std[0]).add_(depth_mean[0])
    
    return torch.clamp(tensors, 0, 10)


def denormalize_rgb(tensor):
    """ Denormalizes image tensors using mean and std """
    
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [1/0.18536709, 1/0.18698769, 1/0.19171533]),
                                transforms.Normalize(mean = [-0.46797212, -0.39178501, -0.37140868],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    inv_tensor = invTrans(tensor)
    torch.round_(inv_tensor.mul_(255.0))
    
    return torch.clamp(inv_tensor, 0, 255)

def createLogger(file_name):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # or any other level
    logger.addHandler(ch)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(file_name, mode='wt')
    fh.setLevel(logging.INFO) # or any level you want
    logger.addHandler(fh)

    fh.setFormatter(formatter)
    
    return logger

def save_my_image(image_array, fp) -> None:
    
    #convering to uint16 -> Grayscale
    _,h,w,c = image_array.shape
    image_array = image_array.reshape(-1,w,c)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # print('image_array',image_array.shape)
    cv2.imwrite(fp, image_array)

def save_sample_images(gt_depth, imgs_rgb, sparse_depth, gen_depth, image_save_path, image_id) -> None:

    denorm_gt = denormalize_dense(gt_depth)
    denorm_sparse = denormalize_sparse(sparse_depth)
    denorm_pred = denormalize_dense(gen_depth)

    gt_depth = generate_depth_cmap(denorm_gt)
    sparse_depth = generate_depth_cmap(denorm_sparse)
    gen_depth = generate_depth_cmap(denorm_pred)

    imgs_rgb = denormalize_rgb(imgs_rgb).permute(0,2,3,1).to('cpu').detach().numpy()
 
    img_grid = np.concatenate((gt_depth, imgs_rgb, sparse_depth, gen_depth), axis=2)
    saved_image_file = os.path.join(image_save_path,"%04d.png"%image_id)
    save_my_image(img_grid, saved_image_file)