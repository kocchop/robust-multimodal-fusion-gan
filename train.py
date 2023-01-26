"""
This is the codebase for Robust Multimodal Fusion GAN paper titled
"Robust Multimodal Depth Estimation using Transformer based Generative Adversarial Networks"
https://dl.acm.org/doi/abs/10.1145/3503161.3548418

Training file for the robust-multimodal-fusion-gan
In order to invoke type:

python train.py --model nyu_modelA --gpus=0,1 --batch_size=40 --n_epochs=27 --decay_epoch=15 --lr_gap=3 -p chkpts/nyu_modelA.pth -n nyu_modelA_train

1. -n --> give a name to the run
2. Modify the val dataloader path with appropriate data directory
3. Typically the directory has the following structure
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

4. The "depth_gt" and "lidar" are the folders containing dense and sparse depth respectively
5. The meta_info.txt contains the file names of these folders. Refer to misc/ folder for sample meta_info file
6. The folder "sample" contains a few sparse samples. This is to track the model learning visually.
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
from validate import validate

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import torch.optim.lr_scheduler as lr_scheduler 

LOGDIR = "./logdir/"

def getOpt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--dataset", type=str, default="nyu_v2", help="name of the dataset (shapeNet or nyu_v2)")
    parser.add_argument("--model", type=str, default="nyu_modelA", required = True, help="name of the model (nyu_modelA | nyu_modelB)")
    parser.add_argument("--dataset_path", type=str, default="/home/dataset/nyu_v2/", help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument('--robust', '-r', action='store_true', help="flag to enable robust training")
    parser.add_argument("--save_size", type=int, default=8, help="batch size for saved outputs")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=15, help="epoch from which to start lr decay")
    parser.add_argument("--lr_gap", type=int, default=4, help="gradient of decay_epoch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_width", type=int, default=304, help="dense depth width")
    parser.add_argument("--channels", type=int, default=1, help="depth image has only 1 channel")
    parser.add_argument("--sample_interval", type=int, default=20, help="interval between saving image samples")
    parser.add_argument("--warmup_batches", type=int, default=250, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--gpus", metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--meta_info_file', '-m', metavar='DIR', default="meta_info.txt", help='Meta file name')
    parser.add_argument("--checkpoint_model_path", type=str, required=False, help="Path to checkpoint model")
    parser.add_argument("--pretrained_model_path", '-p', type=str, required=False, help="Path to pretrained model")

    return parser.parse_args()
        
def main():
    
    # setting higher values initially
    best_rmse = 9999
    best_rel = 9999
    
    opt = getOpt()

    # create the logdir if it does not exists
    os.makedirs(LOGDIR, exist_ok=True)
    
    # create addition log directories
    val_image_save_path = os.path.join(LOGDIR,opt.name,"val_images")
    saved_model_path =  os.path.join(LOGDIR,opt.name,"saved_models")
    log_file_name = os.path.join(LOGDIR,opt.name,'%s.log'%opt.name)
    tensorboard_save_path = os.path.join(LOGDIR,opt.name)

    # os.makedirs(train_image_save_path, exist_ok=True)
    os.makedirs(val_image_save_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

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
    
    if opt.resume_epoch == 0 and opt.pretrained_model_path:
        generator.load_state_dict(torch.load(opt.pretrained_model_path))
    generator = nn.DataParallel(generator, device_ids = opt.gpus)
    generator.cuda()

    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    discriminator = nn.DataParallel(discriminator, device_ids = opt.gpus)
    discriminator.cuda()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_content = NormalLoss().cuda()
    criterion_pixel = torch.nn.L1Loss().cuda()

    if opt.resume_epoch != 0:
        # Load pretrained models
        saved_generator_chkpt = os.path.join(opt.checkpoint_model_path,"generator_%d.pth" % (opt.resume_epoch-1))
        # saved_generator_chkpt = os.path.join(opt.checkpoint_model_path,"generator_best.pth")
        generator.load_state_dict(torch.load(saved_generator_chkpt))
        saved_discriminator_chkpt = os.path.join(opt.checkpoint_model_path,"discriminator_%d.pth" % (opt.resume_epoch-1))
        # saved_discriminator_chkpt = os.path.join(opt.checkpoint_model_path,"discriminator_best.pth")
        discriminator.load_state_dict(torch.load(saved_discriminator_chkpt))
        logger.info("Loaded Checkpoint model from epoch %d"%(opt.resume_epoch-1))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    train_path = os.path.join(opt.dataset_path, "train")
    val_path = os.path.join(opt.dataset_path, "val")

    ## Need to use PairedImageDataset Dataset class
    train_dataloader = DataLoader(
        PairedImageDataset(train_path, opt, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    
    val_dataloader = DataLoader(
        PairedImageDataset(val_path, opt, hr_shape=hr_shape),
        batch_size=opt.save_size,
        num_workers=opt.n_cpu,
    )
    
    # learning rate modification steps
    milestones = [opt.decay_epoch, opt.decay_epoch + opt.lr_gap, opt.decay_epoch + opt.lr_gap*2, opt.decay_epoch + opt.lr_gap*3]
    
    total_train_batches = len(train_dataloader)
    snapshot_interval = round(total_train_batches/2)

    if opt.robust:
        # Finding noisy batches
        train_rgb_noise, train_sparse_noise = send_noisy_batches(total_train_batches, train_flag=True)

        logger.info("RGB noisy batches for training are: {}".format(train_rgb_noise))
        logger.info("Sparse noisy batches for training are: {}".format(train_sparse_noise))

    # ----------
    #  Training
    # ----------
    for epoch in range(opt.resume_epoch, opt.n_epochs):
    
        epoch_start_time = time.time()
        
        # Adjust LR
        if epoch in milestones:
            optimizer_G.param_groups[0]['lr'] *= 0.5 
            optimizer_D.param_groups[0]['lr'] *= 0.5 
        
        for i, imgs in enumerate(train_dataloader):
            
            batches_done = epoch * total_train_batches + i + 1
            
            # this will add channel axis: (Batch Size, Height, Width) --> (Batch Size, 1, Height, Width)
            sparse_temp = torch.unsqueeze(imgs["sparse"], 1)
            gt_temp = torch.unsqueeze(imgs["gt"], 1)
            rgb_temp = imgs["rgb"]
            
            if opt.robust:
                # do not want to start the training during warm up
                rstart = False
                if batches_done >= opt.warmup_batches:
                    rstart = True 
                    
                if (i in train_rgb_noise) and rstart:
                    rgb_temp = torch.zeros(rgb_temp.size()) # it can be any other noise
                    logger.info("Current batch {} is a noisy RGB sample!".format(batches_done))
                elif (i in train_sparse_noise) and rstart:
                    sparse_temp = torch.zeros(sparse_temp.size()) # it can be any other form of noise
                    logger.info("Current batch {} is a noisy sparse sample!".format(batches_done))
            
            # Configure model input
            sparse_depth = Variable(sparse_temp.type(Tensor))
            gt_depth = Variable(gt_temp.type(Tensor))
            imgs_rgb = Variable(rgb_temp.type(Tensor))
            
            #send equal batch partitions to differnt gpus
            sparse_depth, gt_depth, imgs_rgb = sparse_depth.to('cuda'), gt_depth.to('cuda'), imgs_rgb.to('cuda')
                    
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_rgb.size(0), *discriminator.module.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_rgb.size(0), *discriminator.module.output_shape))), requires_grad=False)
            
            valid, fake = valid.to('cuda'), fake.to('cuda')
            
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Construct a depth map using RGB and lidar data
            gen_depth = generator(imgs_rgb, sparse_depth)

            if "nyu" in opt.model:
                gen_depth = gen_depth[:,:,6:-6,:]
                gt_depth = gt_depth[:,:,6:-6,:]

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_depth, gt_depth)
            writer.add_scalar("Pixel_Loss/Train", loss_pixel, batches_done)
            
            # log learning rate
            gen_lr = optimizer_G.param_groups[0]['lr']
            writer.add_scalar("Generateor_LR", gen_lr, batches_done)
            
            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs-1, i+1, len(train_dataloader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(gt_depth).detach()
            pred_fake = discriminator(gen_depth)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            writer.add_scalar("GAN_Loss/Train", loss_GAN, batches_done)

            gen_features = imgrad_yx(gen_depth)
            real_features = imgrad_yx(gt_depth).detach()
            loss_content = criterion_content(gen_features, real_features)
            writer.add_scalar("Content_Loss/Train", loss_content, batches_done)

            # Total generator loss
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
            writer.add_scalar("Generator_Loss/Train", loss_G, batches_done)
            # loss_G = opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(gt_depth)
            pred_fake = discriminator(gen_depth.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            writer.add_scalar("Discriminator_RealLoss/Train", loss_real, batches_done)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
            writer.add_scalar("Discriminator_FakeLoss/Train", loss_fake, batches_done)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            writer.add_scalar("Discriminator_Loss/Train", loss_D, batches_done)
            
            #Discriminator LR
            disc_lr = optimizer_D.param_groups[0]['lr']
            writer.add_scalar("Discriminator_LR", disc_lr, batches_done)
            
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f, lr: %f]" #removed content loss
                % (
                    epoch,
                    opt.n_epochs-1,
                    i+1,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(), # No content loss
                    loss_GAN.item(),
                    loss_pixel.item(),
                    gen_lr,
                )
            )
                
            if batches_done % snapshot_interval == 0:
                # Save model checkpoints
                generator_chkpt = os.path.join(saved_model_path,"generator_%d.pth" % epoch)
                torch.save(generator.state_dict(), generator_chkpt)
                discriminator_chkpt = os.path.join(saved_model_path,"discriminator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), discriminator_chkpt)
                logger.info("Saved Checkpoint at batch {}...".format(batches_done))
                
            
            if batches_done % snapshot_interval == 0:
                with torch.no_grad():
                    avg_rmse, avg_rel = validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done)
                
                # save best checkpoint
                if avg_rmse<best_rmse and avg_rel<best_rel:
                    generator_chkpt = os.path.join(saved_model_path,"generator_best.pth")
                    torch.save(generator.state_dict(), generator_chkpt)
                    discriminator_chkpt = os.path.join(saved_model_path,"discriminator_best.pth")
                    torch.save(discriminator.state_dict(), discriminator_chkpt)
                    logger.info("Saved Best Checkpoint at batch {}...".format(batches_done))
                    best_rel = avg_rel
                    best_rmse = avg_rmse
        
        logger.info("The last epoch took {} hrs... ok.".format((time.time()-epoch_start_time)/3600.0))
    
    # final validation
    with torch.no_grad():
        avg_rmse, avg_rel = validate(generator, discriminator, opt, Tensor, val_dataloader, criterion_GAN, criterion_content, criterion_pixel, logger, val_image_save_path, writer, batches_done)
    
    if avg_rmse<best_rmse and avg_err<best_rel:
        generator_chkpt = os.path.join(saved_model_path,"generator_best.pth")
        torch.save(generator.state_dict(), generator_chkpt)
        discriminator_chkpt = os.path.join(saved_model_path,"discriminator_best.pth")
        torch.save(discriminator.state_dict(), discriminator_chkpt)
        logger.info("Saved Best Checkpoint at batch {}...".format(batches_done))    
    
    writer.flush()
    writer.close()
    
    logger.info("Training Done! Check results.. Adios!")

if __name__=='__main__':
    main()