import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from datasets.humans36m import LoadHumansDataset
from modules.networks import ConditionalImageGenerator 
from sync_batchnorm import DataParallelWithCallback
from modules.losses import l2_loss, discriminator_loss 
from torch.optim.lr_scheduler import MultiStepLR
from modules.vgg19 import Vgg19
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw

def flatten_views(img):
    return img.view(-1, *img.shape[2:])


class ConditionalGenerator2D(nn.Module):

    """
    Full generator model for training and gpu efficiency
    """
    def __init__(self, generator, train_params):
        super(ConditionalGenerator2D, self).__init__()
        self.conditional_generator = generator
        self.vgg = Vgg19().cuda()
        self.loss_weights = train_params['loss_weights']

    def perceptual_loss(self, img_gt, img_generated):
        x_vgg = self.vgg(img_generated)
        y_vgg = self.vgg(img_gt)
        total_value = 0
        for i, weight in enumerate(self.loss_weights['perceptual']):
            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            total_value += value * weight

        return total_value

    def forward(self, img, ref_img, gt_skl):
        generated_image = self.conditional_generator(ref_img, gt_skl)
        loss = l2_loss(img, generated_image)
        out = {}
        out['loss'] = loss
        out['perceptual_loss'] = self.perceptual_loss(img, generated_image)
        out['reconstructred_image'] = generated_image
        return out


def split_data(data, train_views, eval_views):
    img_, kps_, ref_img_ = data
    training_views_idx = torch.LongTensor(train_views)
    eval_views_idx = torch.LongTensor(eval_views)
    image_size = (ref_img_.shape[3] - 1)

    out = {}
    imgs = torch.index_select(img_, 1, training_views_idx)
    imgs_eval = torch.index_select(img_, 1, eval_views_idx)
    out['imgs'] = flatten_views(imgs)
    out['imgs_eval'] = flatten_views(imgs_eval)

    kps = torch.index_select(kps_, 1, training_views_idx) / image_size
    kps_eval = torch.index_select(kps_, 1, eval_views_idx) / image_size
    out['kps'] = flatten_views(kps)
    out['kps_eval'] = flatten_views(kps_eval)

    ref_imgs = torch.index_select(ref_img_, 1, training_views_idx)
    ref_imgs_evaluation = torch.index_select(ref_img_, 1, eval_views_idx)
    out['ref_imgs'] = flatten_views(ref_imgs)
    out['ref_imgs_eval'] = flatten_views(ref_imgs_evaluation)#torch.rand(*out['imgs_eval'].shape)#
    return out

def debug_generator(generator, kp_to_skl_gt, loader, train_params, 
                     logger, device_ids, tgt_batch=None):
    log_params = train_params['log_params']
    genModel = ConditionalGenerator2D(generator, train_params)
    genModel = DataParallelWithCallback(genModel, device_ids=device_ids)

    optimizer_generator = torch.optim.Adam(generator.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    scheduler_generator = MultiStepLR(optimizer_generator, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=-1)
 
    k=0
    train_views = [0,1,3]
    eval_views = [2]
    if tgt_batch is not None:
        tgt_batch_samples = split_data(tgt_batch, 
                                       train_views=train_views, 
                                       eval_views=eval_views)
        with torch.no_grad():
            tgt_batch_samples['gt_skl'] = kp_to_skl_gt(tgt_batch_samples['kps'].to('cuda')).unsqueeze(1)
            tgt_batch_samples['gt_skl_eval'] = kp_to_skl_gt(tgt_batch_samples['kps_eval'].to('cuda')).unsqueeze(1)
        
    for epoch in range(train_params['num_epochs']):
        for i, batch  in enumerate(tqdm(loader)):
            batch_samples = split_data((img, annots, ref_img), 
                                         train_views=train_views, 
                                         eval_views=eval_views)
            #imgs = flatten_views(imgs)
            #ref_imgs = flatten_views(ref_imgs)
            #ref_imgs = torch.rand(*imgs.shape)
            with torch.no_grad():
                batch_samples['gt_skl'] = kp_to_skl_gt(batch_samples['kps'].to('cuda')).unsqueeze(1)
                #batch_samples['gt_skl_eval'] = kp_to_skl_gt(batch_samples['kps_eval'].to('cuda')).unsqueeze(1)
                #gt_skl = (kp_to_skl_gt(flatten_views(annots / (ref_img.shape[3] - 1)).to('cuda'))).unsqueeze(1)
            #gt_skl = torch.rand(imgs.shape[0], 1, *imgs.shape[2:])

            #generator_out = genModel(imgs, ref_imgs, gt_skl)
            generator_out = genModel(batch_samples['imgs'], batch_samples['ref_imgs'], batch_samples['gt_skl'])
            ##### Generator update
            #loss_generator = generator_out['loss']
            loss_generator = generator_out['perceptual_loss']
            loss_generator = [x.mean() for x in loss_generator]
            loss_gen = sum(loss_generator)
            loss_gen.backward(retain_graph=not train_params['detach_kp_discriminator'])
            optimizer_generator.step()
            optimizer_generator.zero_grad()

            ########### LOG
            logger.add_scalar("Generator Loss", 
                               loss_gen.item(), 
                               epoch * len(loader) + i + 1)
            if i in log_params['log_imgs']:
                if tgt_batch is not None:
                    with torch.no_grad():
                        genModel.eval()
                        generator_out_eval = genModel(tgt_batch_samples['imgs_eval'], 
                                                      tgt_batch_samples['ref_imgs_eval'],
                                                      tgt_batch_samples['gt_skl_eval'])
                        #generator_out_eval = genModel(batch_samples['imgs_eval'], 
                        #                              batch_samples['ref_imgs_eval'],
                        #                              batch_samples['gt_skl_eval'])
                        concat_img_eval = np.concatenate((tensor_to_image(tgt_batch_samples['imgs_eval'][k]), 
                                     tensor_to_image(tgt_batch_samples['gt_skl_eval'][k]), 
                                     tensor_to_image(tgt_batch_samples['ref_imgs_eval'][k]),
                                     tensor_to_image(generator_out_eval['reconstructred_image'][k])), axis=2)  # concat along width
                        logger.add_image('Sample_{%d}_EVAL' % i, concat_img_eval, epoch)
                        genModel.train()
                k += 1
                k = k % 4
                concat_img = np.concatenate((tensor_to_image(batch_samples['imgs'][k]), 
                             tensor_to_image(batch_samples['gt_skl'][k]), 
                             tensor_to_image(batch_samples['ref_imgs'][k]),
                             tensor_to_image(generator_out['reconstructred_image'][k])), axis=2)  # concat along width
                logger.add_image('Sample_{%d}' % i, concat_img, epoch)


        scheduler_generator.step()

def tensor_to_image(x):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    out = (out * 255).astype(np.uint8)
    return out
