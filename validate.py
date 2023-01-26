"""
Standalone validation script for the robust-multimodal-fusion-gan
In order to invoke type:

python validate.py --model nyu_modelA --gpus=0 --batch_size=16 --checkpoint_model=./logdir/nyu_train/saved_models/ -n nyu_test

1. The checkpoint model path has to have 2 files named generator_best.pth and discriminator_best.pth
2. -n --> give a name to the run
3. Modify the val dataloader path with appropriate data directory
4. Typically the directory has the following structure
   ----|->data.nyu_v2|
                     |->train|
                             |->sparse_depth
                             |->depth_gt
                             |->image_rgb
                             |->meta_info.txt
                     |->val|
                           |->sparse_depth
                           |->depth_gt
                           |->image_rgb
                           |->meta_info.txt
                     |->sample|
                              |->sparse_depth
                              |->depth_gt
                              |->image_rgb
                              |->meta_info.txt

5. The "depth_gt" and "lidar" are the folders containing dense and sparse depth respectively
6. The meta_info.txt contains the file names of these folders. Refer to misc/ folder for sample meta_info file
7. The folder "sample" contains a few sparse samples. This is to track the model learning visually.
"""
import argparse
import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.generator_models import *
from models.models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

LOGDIR = "./logdir/"

def getOpt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nyu_v2", help="name of the dataset (shapeNet or nyu_v2)")
    parser.add_argument("--model", type=str, default="nyu_modelA", required = True, help="name of the model (nyu_modelA | nyu_modelB)")
    parser.add_argument("--dataset_path", type=str, default="/home/mdl/mzk591/dataset/data.nyuv2/disk3/", help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--save_size", type=int, default=8, help="batch size for saved outputs")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--gpus", metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--meta_info_file', '-m', metavar='DIR', default="meta_info.txt", help='Meta file name')
    parser.add_argument("--checkpoint_model_path", type=str, required=True, help="Path to checkpoint model")

    return parser.parse_args()

def validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done=0):
    
    total_val_batches = len(val_dataloader)
    
    batch_to_be_saved = np.random.randint(total_val_batches, size=5)
    # batch_to_be_saved = [1, 2, 3, 4] #it can be any numbers
    
    val_sample_path = os.path.join(val_image_save_path,"%06d"%batches_done)
    os.makedirs(val_sample_path, exist_ok=True)
    
    loss_dict = {'rmse':[],'rel':[], 'mae':[]}

    for i, imgs in enumerate(val_dataloader):
        
        # this will add channel axis: (Batch Size, Height, Width) --> (Batch Size, 1, Height, Width)
        sparse_temp = torch.unsqueeze(imgs["sparse"], 1)
        gt_temp = torch.unsqueeze(imgs["gt"], 1)
        rgb_temp = imgs["rgb"]
        
        # Configure model input
        sparse_depth = Variable(sparse_temp.type(Tensor))
        gt_depth = Variable(gt_temp.type(Tensor))
        imgs_rgb = Variable(rgb_temp.type(Tensor))
        
        #send equal batch partitions to differnt gpus
        sparse_depth, gt_depth, imgs_rgb = sparse_depth.to('cuda'), gt_depth.to('cuda'), imgs_rgb.to('cuda')
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_rgb.size(0), *discriminator.module.output_shape))), requires_grad=False)
        
        gen_depth = generator(imgs_rgb, sparse_depth)
        
        if "nyu" in opt.model:
                gen_depth = gen_depth[:,:,6:-6,:]
                gt_depth = gt_depth[:,:,6:-6,:]
                sparse_depth = sparse_depth[:,:,6:-6,:]
                imgs_rgb = imgs_rgb[:,:,6:-6,:]

        '''calculation of content, pixel and GAN loss is optional'''

        # Extract validity predictions from discriminator
        pred_real = discriminator(gt_depth).detach()
        pred_fake = discriminator(gen_depth)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        
        gen_features = imgrad_yx(gen_depth)
        real_features = imgrad_yx(gt_depth).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_depth, gt_depth)
        
        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        
        '''calculation of content, pixel and GAN loss is optional'''

        pred, gt = gen_depth.detach().clone(), gt_depth.detach().clone()
        
        pred = denormalize_dense(pred)
        gt = denormalize_dense(gt)

        #new loss measures
        loss_rmse, loss_rel, loss_mae = get_loss(pred, gt)
        
        loss_dict['rmse'].append(loss_rmse.item())
        loss_dict['rel'].append(loss_rel.item())        
        loss_dict['mae'].append(loss_mae.item())
                
        logger.info(
                "Validating [Batch %d/%d] [content: %.3f, pixel: %.3f, RMSE: %.3f, REL: %.3f, MAE: %.3f]" #removed content loss
                % (
                    i+1,
                    len(val_dataloader),
                    loss_content.item(), # No content loss
                    loss_pixel.item(),
                    loss_rmse.item(),
                    loss_rel.item(),
                    loss_mae.item(),
                )
            )
        
        if i in batch_to_be_saved:
            
            save_sample_images(gt_depth, imgs_rgb, sparse_depth, gen_depth, val_sample_path, i)
            logger.info("Saved Validation Images...")
    
    avg_rmse = np.sqrt(np.mean(np.square(loss_dict['rmse'])))
    avg_rel = np.mean(loss_dict['rel'])
    avg_mae = np.mean(loss_dict['mae'])    
    
    writer.add_scalar("Final_RMSE_mean", avg_rmse, batches_done)
    writer.add_scalar("Final_REL_mean", avg_rmse, batches_done)
    writer.add_scalar("Final_MAE_mean", avg_mae, batches_done)
    
    logger.info(
                "Final Avg loss after %d batches [RMSE: %.3f, REL: %.3f, MAE: %.3f]" #removed content loss
                % (
                    batches_done,
                    avg_rmse,
                    avg_rel,
                    avg_mae,
                )
            )
    
    return avg_rmse, avg_rel
        
def main():
    
    opt = getOpt()

    # create the logdir if it does not exist
    os.makedirs(LOGDIR, exist_ok=True)
   
    val_image_save_path = os.path.join(LOGDIR,opt.name,"val_images")
    log_file_name = os.path.join(LOGDIR,opt.name,'%s.log'%opt.name)
    tensorboard_save_path = os.path.join(LOGDIR,opt.name)

    os.makedirs(val_image_save_path, exist_ok=True)

    # Create a logger
    logger = createLogger(log_file_name)

    # print(opt)
    logger.info(opt)
    
    # initiate tensorboard logger
    writer = SummaryWriter(log_dir=tensorboard_save_path)
    

    if opt.gpus is not None:
        try:
            opt.gpus = [int(s) for s in opt.gpus.split(',')]
        except ValueError:
            logger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in opt.gpus:
            if dev_id >= available_gpus:
                logger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(opt.gpus[0])


    if 'shapeNet' in opt.model:
        hr_shape = (192, 256)
    elif "nyu" in opt.model:
        hr_shape = (240, 304)
    
    model_config = {
            "img_size": hr_shape,
            "rgb_chans": 3,
            "lidar_chans": 1,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
    }

    # Initialize generator and discriminator
    try:
        generator = eval(opt.model)(**model_config)
    except:
        print("Please select model from: nyu_modelA | nyu_modelB")
        quit()
    
    generator = nn.DataParallel(generator, device_ids = opt.gpus)
    generator.cuda()

    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    discriminator = nn.DataParallel(discriminator, device_ids = opt.gpus)
    discriminator.cuda()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_content = NormalLoss().cuda()
    criterion_pixel = torch.nn.L1Loss().cuda()

    # Load state dict for generator and discriminator
    saved_generator_chkpt = os.path.join(opt.checkpoint_model_path,"generator_best.pth")
    generator.load_state_dict(torch.load(saved_generator_chkpt))
    saved_discriminator_chkpt = os.path.join(opt.checkpoint_model_path,"discriminator_best.pth")
    discriminator.load_state_dict(torch.load(saved_discriminator_chkpt))
    
    # Only evaluate
    generator.eval()
    discriminator.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    
    val_path = os.path.join(opt.dataset_path, "val")

    ## Need to use PairedImageDataset Dataset class    
    val_dataloader = DataLoader(
        PairedImageDataset(val_path, opt, hr_shape=hr_shape),
        batch_size=opt.save_size,
        num_workers=opt.n_cpu,
    )

    # final validation
    with torch.no_grad():
        avg_rmse, avg_rel =validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer)
    
    writer.flush()
    writer.close()
    
    logger.info("Validation Done. Check results!")

if __name__=='__main__':
    main()