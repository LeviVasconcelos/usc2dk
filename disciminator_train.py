import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from evaluation import evaluate
from sync_batchnorm import DataParallelWithCallback 
from modules.losses import masked_l2_heatmap_loss, l1_loss, masked_l2_loss 
from modules.losses import generator_gan_loss, discriminator_gan_loss, mean_batch
from modules.util import kp2gaussian2
from modules.util import batch_image_rotation, batch_kp_rotation, discriminator_percentile 
from datasets.annot_converter import HUMANS_TO_HUMANS
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ColorJitter, ToPILImage
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw
import random
from modules.affine_augmentation import batch_kp_affine, batch_img_affine, inverse_aff_values, inverse_affine, batch_affine



def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

def gaussian2kp(heatmap):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    #adding small eps to avoid 'nan' in variance
    kp = {}
    kp['var'] = torch.var(heatmap,(2,3))
    heatmap = heatmap.unsqueeze(-1) + 1e-7
    grid_ = make_coordinate_grid(shape[2:], heatmap.type())
    grid = grid_.unsqueeze(0).unsqueeze(0)
    grid =  grid.to(heatmap.device)
    mean_ = (heatmap * grid)
    mean = mean_.sum(dim=(2, 3))

    kp['mean'] =  mean 

    return kp  

class DiscriminatorTrainer(nn.Module):
    def __init__(self, discriminator, train_params):
        super(DiscriminatorTrainer, self).__init__()
        self.discriminator = discriminator
        self.loss_weights = train_params['loss_weights']

    def forward(self, gt_images, generated_images, filter=None):
        gt_maps = self.discriminator(gt_images.detach())
        gen_maps = self.discriminator(generated_images.detach())
        disc_loss = []
        for i in range(len(gt_maps)):
            gen_map = gen_maps[i]
            gt_map = gt_maps[i]
            disc_loss.append(discriminator_gan_loss(
                               discriminator_maps_generated=gen_map,
                               discriminator_maps_real=gt_map,
                               weight=self.loss_weights['discriminator_gan']).mean())
        return {
                'loss': (sum(disc_loss) / len(disc_loss)),
                'scales': disc_loss,
               }

def train_discriminator(label_generator,
                    model_discriminator,
                    loaders,
                    train_params,
                    checkpoint,
                    logger, device_ids, 
                    kp_map=None):

    log_params = train_params['log_params']
    resume_epoch = 0
    resume_iteration = 0

    optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, 
                                    train_params['epoch_milestones'], 
                                    gamma=0.1, last_epoch=logger.epoch-1)

    if checkpoint is not None:
        print('Loading Checkpoint: %s' % checkpoint)
        if train_params['test'] == False:
            resume_epoch, resume_iteration = logger.checkpoint.load_checkpoint(checkpoint,
                                                  model_kp_detector=model_kp_detector,
                                                  optimizer_kp_detector=optimizer_kp_detector)
        logger.epoch = resume_epoch
        logger.iterations = resume_iteration
    
    discriminator = DiscriminatorTrainer(model_discriminator, train_params)

    k = 0

    heatmap_var = train_params['heatmap_var']
    heatmap_size = (train_params['heatmap_res'],train_params['heatmap_res'])
    loader_src_train, loader_src_test, loader_tgt = loaders
    iterator_source = iter(loader_src_test)
    device_ids =device_ids[0]


    for epoch in range(logger.epoch, train_params['num_epochs']):
        if epoch % 20 == 0:
            logger.save_model(models = {'model_kp_discriminator':model_discriminator,
                        'optimizer_kp_discriminator':optimizer_discriminator}) 
            print(f"saved discriminator")   
        
        for i, src_batch_train  in enumerate(tqdm(loader_src_train)):
            src_images_train = src_batch_train['imgs'].to(device_ids)
            src_annots_train = src_batch_train['annots'].to(device_ids)
            src_annots_hm_train = kp2gaussian2(src_annots_train, heatmap_size, heatmap_var)

            mask_src_train = None if 'kp_mask' not in src_batch_train.keys() else src_batch_train['kp_mask']
        
            try:
                src_batch_test = next(iterator_source)
            except:
                iterator_source = iter(loader_src_test)
                src_batch = next(iterator_source)

            src_images_test = src_batch_test['imgs'].to(device_ids)
            src_annots_test = src_batch_test['annots'].to(device_ids)
            src_annots_hm_test = kp2gaussian2(src_annots_test, heatmap_size, heatmap_var)

            mask_src_test = None if 'kp_mask' not in src_batch_test.keys() else src_batch_test['kp_mask']

            with torch.no_grad():
                dict_out_source = label_generator(src_images_train)
            disc_loss_value = 0

            hm = dict_out_source['heatmaps'].detach()
            kp = gaussian2kp(hm)["mean"]
            hm = kp2gaussian2(kp, heatmap_size, heatmap_var).detach()

            disc_out = discriminator(src_annots_hm_train.detach(), 
                                    hm, filter=kp_map)

            disc_loss = disc_out['loss'].mean()
            disc_loss_value = disc_loss.item()
            optimizer_discriminator.zero_grad()
            disc_loss.backward()
            optimizer_discriminator.step()


            ####### LOG
            logger.add_scalar('disc loss',
                               disc_loss_value,
                               logger.iterations)
            logger.step_it()

        scheduler_discriminator.step()


def draw_kp(img_, kps, color='blue'):
    #print(img_.shape)
    img = img_.transpose(1,2,0) if img_.shape[0] == 3 else img_
    img = Image.fromarray(img)
    kp_img = img.copy()
    draw = ImageDraw.Draw(kp_img)
    radius = 2
    for kp in kps:
        rect = [kp[0] - radius, kp[1] - radius, kp[0] + radius, kp[1] + radius]
        draw.ellipse(rect, fill=color, outline=color)
    return np.array(kp_img).transpose(2,0,1)

def unnorm_kp(kps):
    return (127./2.) * (kps + 1)

def tensor_to_image(x, should_normalize=False):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    if should_normalize:
        max_value = np.max(out)
        min_value = np.min(out)
        out = (out - min_value) / (max_value - min_value)
    out = (out * 255).astype(np.uint8)
    return out


