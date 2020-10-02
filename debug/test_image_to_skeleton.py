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
from sync_batchnorm import DataParallelWithCallback
from modules.losses import l2_loss, discriminator_gan_loss_v2, generator_loss_2d, generator_gan_loss_v2
from modules.networks import Discriminator2D, MultiScaleDiscriminator
from torch.optim.lr_scheduler import MultiStepLR
from modules.vgg19 import Vgg19
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw


def flatten_views(img):
    return img.view(-1, *img.shape[2:])

class SkeletonGenerator(nn.Module):
    """
    Skeleton generator model to debug
    """
    def __init__(self, img_to_skl, discriminators, train_params):
        super(SkeletonGenerator, self).__init__()
        self.img_to_skl = img_to_skl
        self.discriminators = discriminators
        self.train_params = train_params
        self.loss_weights = train_params['loss_weights']

    def forward(self, img):
        generated_skeletons = self.img_to_skl(img) 
        #print('img range: ' + str(img.min()) + ',' + str(img.max()))
        #print('generated range: ' + str(generated_skeletons.min()) + ',' + str(generated_skeletons.max()))
        gen_predictions_fakes = self.discriminators(generated_skeletons)
        prediction_reals = None 

        gen_loss = [] 
        for map_generated in gen_predictions_fakes:
            gen_loss.append(generator_gan_loss_v2(discriminator_maps_generated=map_generated,
                                      weight=self.train_params['loss_weights']['generator_gan']).mean())
            # gen_loss has shape [bsz]
 
        out = {}
        out['skeleton_image'] = generated_skeletons
        out['gen_loss'] = sum(gen_loss)/len(gen_loss) 
        return out 

class SkeletonDiscriminator(nn.Module):
    """
    Skeleton discriminator 
    """
    def __init__(self, discriminators, train_params):
        super(SkeletonDiscriminator, self).__init__()
        self.discriminators = discriminators
        self.train_params = train_params

    def forward(self, gt_skeletons, generated_skeletons):
        gt_maps = self.discriminators(gt_skeletons)
        generated_maps = self.discriminators(generated_skeletons.detach())
        disc_loss = [] 
        for i in range(len(gt_maps)):
            generated_map = generated_maps[i]
            gt_map = gt_maps[i]
            #print('[SkeletonDiscriminator] gt_map shape: ', gt_map.shape)
            disc_loss.append(discriminator_gan_loss_v2(discriminator_maps_generated=generated_map,
                                      discriminator_maps_real=gt_map,
                                      weight=self.train_params['loss_weights']['discriminator_gan']).mean())
        return (sum(disc_loss) / len(gt_maps))
 
def debug_skeleton_prediction(img_to_skl, discriminators, kp_to_skl_gt, 
                               loader, train_params, 
                               logger, device_ids):
     log_params = train_params['log_params']
     generatorModel = SkeletonGenerator(img_to_skl, discriminators, train_params)
     generatorModel = DataParallelWithCallback(generatorModel, device_ids=device_ids)
     
     discriminatorModel = SkeletonDiscriminator(discriminators, train_params)
     discriminatorModel = DataParallelWithCallback(discriminatorModel, device_ids=device_ids)

     optimizer_image_to_skeleton = torch.optim.Adam(img_to_skl.parameters(),
                                                     lr=train_params['lr'],
                                                     betas=train_params['betas'])
     scheduler_image_to_skeleton = MultiStepLR(optimizer_image_to_skeleton, 
                                                train_params['epoch_milestones'], 
                                                gamma=0.1, last_epoch=-1)
 
     optimizer_discriminator = torch.optim.Adam(discriminators.parameters(),
                                                     lr=train_params['lr'],
                                                     betas=train_params['betas'])
     scheduler_discriminator = MultiStepLR(optimizer_discriminator, 
                                                train_params['epoch_milestones'], 
                                                gamma=0.1, last_epoch=-1)

     SkeletonFromKP = lambda x : kp_to_skl_gt(x).to('cuda').unsqueeze(1)
     for epoch in range(train_params['num_epochs']):
         loader.dataset.shuffle_unaligned()
         for i,batch in enumerate(tqdm(loader)):
             with torch.no_grad():
                 gt_skl_unaliagned = SkeletonFromKP(batch['unaligned_annots'])
                 gt_skl_aligned = SkeletonFromKP(batch['annots'])

             gt_to_use = gt_skl_unaliagned

            
             generated_out = generatorModel(img=batch['imgs'])
             # gen_loss has size [batch, gpu]
             #loss_generator = [x.mean() for x in generated_out['gen_loss']]
             loss_gen = generated_out['gen_loss'].mean()
             loss_gen.backward()

             if train_params['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(generatorModel.parameters(), train_params['max_grad_norm'])
             gen_grad = img_to_skl.decoder.final_conv.weight.grad.data.norm(2).item()
             optimizer_image_to_skeleton.step()
             optimizer_image_to_skeleton.zero_grad()
             optimizer_discriminator.zero_grad() # This is necessary to not modify the discriminator

             discriminator_out = discriminatorModel(gt_skeletons=gt_to_use.detach(), 
                                                     generated_skeletons=generated_out['skeleton_image'].detach()) 
 
             #loss_discriminator = [x.mean() for x in discriminator_out]
             loss_disc = discriminator_out.mean()
             loss_disc.backward()
             if train_params['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(discriminatorModel.parameters(), train_params['max_grad_norm'])
             disc_grads = list()
             for d in discriminators.models:
                 disc_grads.append(d.conv.weight.grad.data.norm(2).item())
             optimizer_discriminator.step()
             optimizer_discriminator.zero_grad()
             optimizer_image_to_skeleton.zero_grad()
              
             logger.add_scalar("Losses", 
                                loss_disc.item() + loss_gen.item(), 
                                logger.iterations)
             logger.add_scalar("Disc Loss", 
                                loss_disc.item(), 
                                logger.iterations)
             logger.add_scalar("Gen Loss", 
                                loss_gen.item(), 
                                logger.iterations)
             logger.add_scalar("Generator Gradient",
                                 gen_grad,
                                 logger.iterations)
             disc_names = ["1","1/2","1/4"]
             for l,legend in zip(disc_grads, disc_names):
                 logger.add_scalar("Discriminators/%s" % legend,
                                    l,
                                    logger.iterations)
                                

             if i in log_params['log_imgs']:
                 concat_img = np.concatenate((tensor_to_image(batch['imgs'][0]), 
                              tensor_to_image(gt_skl_aligned[0]),
                              tensor_to_image(gt_to_use[0]), 
                              tensor_to_image(generated_out['skeleton_image'][0])), axis=2)
                 logger.add_image('Sample_{%d}' % i, concat_img, epoch)
             logger.step_it()
             
         scheduler_image_to_skeleton.step()
         scheduler_discriminator.step()

def tensor_to_image(x):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    out = (out * 255).astype(np.uint8)
    return out
