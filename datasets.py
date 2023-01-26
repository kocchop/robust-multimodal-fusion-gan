"""
This is sample dataloader script for robust multimodal fusion GAN
This dataloader script is for nyu_v2 dataset where
The sparse depth and ground truth depths are stored as h5 file, and
The rgb image is stored as a png
"""
import glob
import random
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import *

import cv2
import h5py
from PIL import Image
import torchvision.transforms as transforms

def read_gt_depth(path):
    
    file = h5py.File(path, "r")
    gt_depth = np.array(file['depth_gt'])
    
    return gt_depth 

def read_sparse_depth(path):
    
    file = h5py.File(path, "r")
    sparse_depth = np.array(file['lidar'])
    
    return sparse_depth 

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
    
def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.
    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.
    Example of an meta information file:
    ```
    0001.png (228,304,1)
    0002.png (228,304,1)
    ```
    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder, rgb_foldar].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt', 'rgb'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [input_folder, gt_folder, rgb_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 2 with [input_key, gt_key, rgb_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder, rgb_folder = folders
    input_key, gt_key, rgb_key = keys
    

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    
    rgb_ext  = '.png'
    depth_ext = '.h5'
    
    for basename in gt_names:
        input_name = f'{filename_tmpl.format(basename)}{depth_ext}'
        rgb_name = f'{filename_tmpl.format(basename)}{rgb_ext}'
        gt_name = f'{filename_tmpl.format(basename)}{depth_ext}'
        
        input_path = os.path.join(input_folder, input_name)
        rgb_path = os.path.join(rgb_folder, rgb_name)
        gt_path = os.path.join(gt_folder, gt_name)

        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path),
                  (f'{rgb_key}_path', rgb_path)]))
    return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.
    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt', 'rgb'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, (
        'The len of folders should be 3 with [input_folder, gt_folder, rgb_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [input_key, gt_key, rgb_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder, rgb_folder = folders
    input_key, gt_key, rgb_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    rgb_paths = list(scandir(rgb_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    assert len(input_paths) == len(rgb_paths), (
        f'{input_key} and {rgb_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(rgb_paths)}.')
    paths = []
    
    rgb_ext = '.png'
    
    for gt_path in gt_paths:
        basename, ext = os.path.splitext(os.path.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        rgb_name = f'{filename_tmpl.format(basename)}{rgb_ext}'

        input_path = os.path.join(input_folder, input_name)
        gt_path = os.path.join(gt_folder, gt_path)
        rgb_path = os.path.join(rgb_folder, rgb_path)
        
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path),
                  (f'{rgb_key}_path', rgb_path)]))
    return paths


class PairedImageDataset(Dataset):

    def __init__(self, root, opt, hr_shape):
    #We cannot use torch.Transforms because transforms.ToTensor() normalizes the image assuming its a 3 channel uint8 RGB image
        super(PairedImageDataset, self).__init__()
        
        self.opt = opt
        
        # assumption is that the sparse depth is in "lidar" folder
        #                   ground truth depth is in "depth_gt" folder
        #               and rgb image is in "image_rgb" folder
        self.gt_folder, self.lq_folder, self.rgb_folder = os.path.join(root,'depth_gt'), os.path.join(root,'sparse_depth'), os.path.join(root,'image_rgb')
        
        self.filename_tmpl = '{}'
        
        self.transform_rgb = transforms.Compose([transforms.Pad((0,6,0,6),fill=0),
                                                 transforms.ToTensor(),                                               
                                                 transforms.Normalize(mean = rgb_mean,
                                                                       std = rgb_std),
                                                                       ])
        
        if self.opt.meta_info_file is not None:
            self.meta_file = os.path.join(root, self.opt.meta_info_file)
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.rgb_folder], ['lq', 'gt', 'rgb'],
                self.meta_file, self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.rgb_folder], ['lq', 'gt', 'rgb'],
                self.filename_tmpl)

    def __getitem__(self, index):

        # Load gt and lq depths. Dimension order: HW; channel: Grayscale;
        # Depth range: [0, 9.999], float32.
        gt_path = self.paths[index]['gt_path']
        img_hi = read_gt_depth(gt_path)
        temp_hi = torch.from_numpy(img_hi)
        img_hi = F.pad(temp_hi,(0,0,6,6),'constant',0)
        
        lq_path = self.paths[index]['lq_path']
        img_lo = read_sparse_depth(lq_path)
        temp_lo = torch.from_numpy(img_lo)
        img_lo = F.pad(temp_lo,(0,0,6,6),'constant',0)

        rgb_path = self.paths[index]['rgb_path']
        img_color = Image.open(rgb_path)
        
        # depth transformation
        gt = (img_hi-depth_mean)/depth_std
        sparse = (img_lo-sparse_mean)/sparse_std
        
        # RGB transformation
        img_rgb = self.transform_rgb(img_color)

        return {
            'sparse': sparse,
            'gt': gt,
            'rgb': img_rgb
        }

    def __len__(self):
        return len(self.paths)

