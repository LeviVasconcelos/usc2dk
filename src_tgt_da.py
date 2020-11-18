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
from modules.losses import generator_gan_loss, discriminator_gan_loss
from modules.util import kp2gaussian2, gaussian2kp
from modules.util import batch_image_rotation, batch_kp_rotation 
from datasets.annot_converter import HUMANS_TO_HUMANS
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw
import copy
import random

class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector, train_params, 
                       discriminator=None, 
                       geo_transform=None,
                       kp_to_skl=None, 
                       angle_equivariance=False):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.detector.convert_bn_to_dial(self.detector)
        self.discriminator = discriminator
        self.heatmap_res = self.detector.heatmap_res
        self.geo_transform = geo_transform
        self.train_params = train_params
        self.loss_weights = self.train_params['loss_weights']
        self.to_skeleton = kp_to_skl
        self.epoch_ratio = 0
        self.angle_equivariance = angle_equivariance
        self.geo_f = torch.nn.MSELoss()
        self.angle_range = 45
        self.angle_incr_factor = 2



    def forward(self, images, ground_truth, mask=None, kp_map=None):
        self.detector.set_domain_all(source=True)
        dict_out = self.detector(images)
        gt = ground_truth if kp_map is None else ground_truth[:, kp_map]
        src_loss = 0
        if self.train_params['source_loss'] == 'kp_values':
            src_loss = masked_l2_loss(dict_out['value'], gt, mask)
        elif self.train_params['source_loss'] == 'heatmap':
            size = (self.heatmap_res, self.heatmap_res)
            variance = self.train_params['heatmap_var']
            gt_heatmaps = kp2gaussian2(gt, size, variance).detach()
            src_loss = masked_l2_heatmap_loss(dict_out['heatmaps'], 
                                               gt_heatmaps, mask)
        heatmaps = dict_out['heatmaps'][:, kp_map] if kp_map is not None else dict_out['heatmaps']
        geo_loss = 0
        if self.geo_transform is not None:
            range_angle = min(self.angle_range ,int(self.angle_incr_factor * (self.angle_range* self.epoch_ratio)))
            angle = random.randint(-1*range_angle,range_angle)
            geo_images = self.geo_transform(images, angle).detach()
            geo_dict_out = self.detector(geo_images)
            geo_heatmaps = geo_dict_out['heatmaps'] if kp_map is None else geo_dict_out['heatmaps'][:, kp_map]
            if self.angle_equivariance:
                geo_loss = self.angle_difference(dict_out['value'],geo_dict_out['value'], angle, device=self.device)
            else:
                geo_loss = self.equivariance_loss(heatmaps, geo_heatmaps, angle)

        kps = unnorm_kp(dict_out['value'])
        return {"keypoints": kps,
                "heatmaps": dict_out['heatmaps'],
                "src_loss": src_loss.mean(),
                "geo_loss": geo_loss,
                }
    def equivariance_loss(self, source, transformed, geo_param):
        forward = l1_loss(self.geo_transform(source, geo_param), 
                           transformed).mean()
        backward = l1_loss(source,
                            self.geo_transform(transformed, -geo_param)).mean()
        return forward + backward
    
    def angle_difference(self,kps,rot_kps,gt_angle, device="cuda"):

        angle = torch.atan2(torch.det(torch.stack([rot_kps,kps], 2)),torch.sum(rot_kps*kps, dim=2))
        gt_angle = torch.tensor(gt_angle).type(torch.FloatTensor)
        gt_angle = gt_angle.to(device)
        angle = (angle*180/np.pi).mean()
        return self.geo_f(angle,gt_angle)

    def adapt(self, images, kp_map):
        self.detector.set_domain_all(source=False)
        dict_out = self.detector(images)
        kps = dict_out['value'][:, kp_map]
        kps = unnorm_kp(kps)
        heatmaps = dict_out['heatmaps'][:, kp_map]
        geo_loss = 0
        geo_dict_out = None
        geo_images = None
        if self.geo_transform is not None:
            range_angle = min(self.angle_range ,int(self.angle_incr_factor * (self.angle_range* self.epoch_ratio)))
            angle = random.randint(-1*range_angle,range_angle)
            geo_images = self.geo_transform(images, angle).detach()
            geo_dict_out = self.detector(geo_images)
            geo_heatmaps = geo_dict_out['heatmaps'][:, kp_map]
            if self.angle_equivariance:
                geo_loss = self.angle_difference(kps,geo_dict_out['value'][:,kp_map], angle, device=self.device)
            else: 
                geo_loss = self.equivariance_loss(heatmaps, geo_heatmaps, angle)
        generator_loss = 0
        if self.discriminator is not None:
            generator_scores = []
            maps = self.discriminator(heatmaps)
            for i, gen_map in enumerate(maps):
                gen_scores = generator_gan_loss(gen_map, 
                                           self.loss_weights['generator_gan']).mean()
                generator_scores.append(gen_scores)
            generator_loss = sum(generator_scores) / len(generator_scores)

        return {"keypoints": kps,
                "heatmaps": heatmaps,
                "geo_loss": geo_loss,
                "geo_out": geo_dict_out,
                "geo_images": geo_images,
                "generator_loss": generator_loss,
                }

class DiscriminatorTrainer(nn.Module):
    def __init__(self, discriminator, train_params):
        super(DiscriminatorTrainer, self).__init__()
        self.discriminator = discriminator
        self.loss_weights = train_params['loss_weights']

    def forward(self, gt_images, generated_images):
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
        

def d_unrolled_loop(d_gen_input=None, real_data=None,optimizer_discriminator=None, gen=None,kp_map=None, discriminator=None, train_params=None):
    
    optimizer_discriminator.zero_grad()
    with torch.no_grad():
        fake_data = gen.adapt(d_gen_input,kp_map)
        
    
    gt_maps = discriminator(real_data)
    generated_maps = discriminator(fake_data['heatmaps'].detach())
    disc_loss = [] 
    for i in range(len(gt_maps)):
        generated_map = generated_maps[i]
        gt_map = gt_maps[i]
        disc_loss.append(discriminator_gan_loss(discriminator_maps_generated=generated_map,
                                    discriminator_maps_real=gt_map,
                                    weight=train_params['loss_weights']['discriminator_gan']).mean())
    out_disc = { 'loss': (sum(disc_loss) / len(disc_loss)),
                  'scales': disc_loss, }

    loss_disc = out_disc['loss'].mean() 
    loss_disc.backward(create_graph=True)

    optimizer_discriminator.step()
    optimizer_discriminator.zero_grad()

def train_kpdetector(model_kp_detector,
                       loaders,
                       train_params,
                       checkpoint,
                       logger, device_ids, model_discriminator=None,
                       kp_map=None):
    log_params = train_params['log_params']
    resume_epoch = 0
    resume_iteration = 0
    optimizer_kp_detector = torch.optim.Adam(model_kp_detector.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)


    if model_discriminator is not None:
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


    geo_transform = None
    if train_params['use_rotation']:
        geo_transform = batch_image_rotation
    kp_detector = KPDetectorTrainer(model_kp_detector, train_params, 
                                    discriminator=model_discriminator,
                                    geo_transform=geo_transform, angle_equivariance=train_params['angle_equivariance'])
    if train_params['use_gan']:
        discriminator = DiscriminatorTrainer(model_discriminator, train_params)
    # number of unrolled steps to do, if 0 no unrolling, usually 5 or 10
    unrolled_steps = train_params["unrolled_steps"] 
    print(f"unrolled_steps: {unrolled_steps}, rotation: {train_params['use_rotation']}, angle_equivariance {train_params['angle_equivariance']}")

    k = 0
    if train_params['test'] == True:
        results = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'])
        print(' MSE: ' + str(results['MSE']) + ' PCK: ' + str(results['PCK'])) 
        return

    heatmap_var = train_params['heatmap_var']
    heatmap_size = (kp_detector.heatmap_res, kp_detector.heatmap_res)
    loader_src_train, loader_src_test, loader_tgt = loaders
    iterator_source = iter(loader_src_train)
    for epoch in range(logger.epoch, train_params['num_epochs']):
        kp_detector.epoch_ratio = epoch/train_params["num_epochs"]
        kp_detector.detector.set_domain_all(source=False)
        results_tgt = evaluate(kp_detector.detector, loader_tgt, dset=train_params['tgt_train'], filter=kp_map)
        kp_detector.detector.set_domain_all(source=True)
        results_src_test = evaluate(kp_detector.detector, loader_src_test, dset=train_params['src_test']) 
        results_src_train = evaluate(kp_detector.detector, loader_src_train, dset=train_params['src_train'])
        print('Epoch ' + str(epoch)+ ' PCK_target: ' + str(results_tgt['PCK']))
        logger.add_scalar('MSE target', results_tgt['MSE'], epoch)
        logger.add_scalar('PCK target', results_tgt['PCK'], epoch)
        logger.add_scalar('MSE src train', results_src_train['MSE'], epoch)
        logger.add_scalar('PCK src train', results_src_train['PCK'], epoch)
        logger.add_scalar('MSE src test', results_src_test['MSE'], epoch)
        logger.add_scalar('PCK src test', results_src_test['PCK'], epoch)
   
        for i, tgt_batch  in enumerate(tqdm(loader_tgt)):
            tgt_images = tgt_batch['imgs'].cuda()
            tgt_annots = tgt_batch['annots'].cuda()
            try:
                src_batch = next(iterator_source)
            except:
                iterator_source = iter(loader_src_train)
                src_batch = next(iterator_source)
            src_images = src_batch['imgs'].cuda()
            src_annots = src_batch['annots'].cuda()
            src_annots_hm = kp2gaussian2(src_annots, heatmap_size, heatmap_var)
            mask = None if 'kp_mask' not in src_batch.keys() else src_batch['kp_mask']
            
            ## code adapted from https://github.com/andrewliao11/unrolled-gans
            ## Unroll the discriminator here making a deep copy
            if train_params['use_gan'] and unrolled_steps > 0:
                backup = copy.deepcopy(discriminator.state_dict())
                for iterat in range(unrolled_steps):
                    # unrolled loop for non rotated frames
                    kp_detector.detector.set_domain_all(source=False)
                    d_unrolled_loop(d_gen_input=tgt_images, real_data=src_annots_hm[:, kp_map].detach(),optimizer_discriminator=optimizer_discriminator, gen=kp_detector, kp_map=kp_map, discriminator=model_discriminator, train_params=train_params)
                    kp_detector.detector.set_domain_all(source=True)

            kp_detector_src_out = kp_detector(src_images, src_annots, mask)
            kp_detector_tgt_out = kp_detector.adapt(tgt_images, kp_map)

            src_loss = kp_detector_src_out['src_loss'].mean() 
            geo_loss_value = 0
            geo_loss = 0
            if train_params['use_rotation']:
                geo_out = kp_detector_tgt_out['geo_out']
                geo_loss = train_params['loss_weights']['geometric'] * \
                       (kp_detector_tgt_out['geo_loss'] + kp_detector_src_out['geo_loss'])
                geo_loss_value = geo_loss.item()

            gen_loss = 0
            gen_loss_value = 0
            if train_params['use_gan']:
                gen_loss = kp_detector_tgt_out['generator_loss']
                gen_loss_value = gen_loss.item()

            loss = geo_loss + src_loss + gen_loss
            loss.backward()

            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()

            # reload discriminator model
            if unrolled_steps > 0:
                discriminator.load_state_dict(backup)

            disc_loss_value = 0
            if train_params['use_gan']:
                disc_out = discriminator(src_annots_hm[:, kp_map].detach(), 
                                         kp_detector_tgt_out['heatmaps'].detach())
                disc_loss = disc_out['loss'].mean()
                disc_loss_value = disc_loss.item()
                optimizer_discriminator.zero_grad()
                disc_loss.backward()
                optimizer_discriminator.step()
                optimizer_kp_detector.zero_grad()


            ####### LOG
            logger.add_scalar('src l2 loss', 
                               src_loss.item(), 
                               logger.iterations)
            logger.add_scalar('tgt geo loss',
                               geo_loss_value,
                               logger.iterations)
            logger.add_scalar('disc loss',
                               disc_loss_value,
                               logger.iterations)
            logger.add_scalar('gen loss',
                               gen_loss_value,
                               logger.iterations)
            
            if i in log_params['log_imgs']:
                concat_img_src = np.concatenate((
                    draw_kp(tensor_to_image(src_images[k]), 
                             unnorm_kp(src_annots[k])),
                    draw_kp(tensor_to_image(src_images[k]), 
                             kp_detector_src_out['keypoints'][k], color='red')),
                             axis=2)
                concat_img_tgt = np.concatenate((
                    draw_kp(tensor_to_image(tgt_images[k]), 
                             unnorm_kp(tgt_annots[k])),
                    draw_kp(tensor_to_image(tgt_images[k]), 
                             kp_detector_tgt_out['keypoints'][k], color='red')),
                             axis=2)
                if train_params['use_rotation']:
                    concat_img_geo = np.concatenate((
                        draw_kp(tensor_to_image(tgt_images[k]), 
                                 kp_detector_tgt_out['keypoints'][k], color='red'),
                        draw_kp(tensor_to_image(kp_detector_tgt_out['geo_images'][k]), 
                                 unnorm_kp(kp_detector_tgt_out['geo_out']['value'][k]), color='red')),
                                 axis=2)

                    logger.add_image('geo tgt', concat_img_geo, logger.iterations)

                heatmap_0 = kp_detector_src_out['heatmaps'][k, 0].unsqueeze(0)
                heatmap_1 = kp_detector_src_out['heatmaps'][k, 1].unsqueeze(0)
                concat_hm_src_net = np.concatenate((
                       tensor_to_image(heatmap_0, True),
                       tensor_to_image(heatmap_1, True)), axis= 2)
                heatmap_0 = kp_detector_tgt_out['heatmaps'][k, 0].unsqueeze(0)
                heatmap_1 = kp_detector_tgt_out['heatmaps'][k, 1].unsqueeze(0)
                concat_hm_tgt_net = np.concatenate((
                       tensor_to_image(heatmap_0, True),
                       tensor_to_image(heatmap_1, True)), axis= 2)
                heatmap_0 = src_annots_hm[k, 0].unsqueeze(0)
                heatmap_1 = src_annots_hm[k, 1].unsqueeze(0)
                concat_hm_gt = np.concatenate((
                       tensor_to_image(heatmap_0, True),
                       tensor_to_image(heatmap_1, True)), axis= 2)
                
                logger.add_image('src train', concat_img_src, logger.iterations)
                logger.add_image('tgt train', concat_img_tgt, logger.iterations)
                logger.add_image('src_heatmap', concat_hm_src_net, logger.iterations)
                logger.add_image('tgt_heatmap', concat_hm_tgt_net, logger.iterations)
                logger.add_image('gt heatmap', concat_hm_gt, logger.iterations)
                k += 1
                k = k % len(log_params['log_imgs']) 
            logger.step_it()

        scheduler_kp_detector.step()
        if train_params['use_gan']:
            scheduler_discriminator.step()

        logger.step_epoch(models = {'model_kp_detector':model_kp_detector,
                                    'optimizer_kp_detector':optimizer_kp_detector})

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


