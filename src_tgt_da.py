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
from modules.util import kp2gaussian2, gaussian2kp
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


class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector, train_params, 
                       discriminator=None, 
                       geo_transform=None,
                       kp_to_skl=None):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.discriminator = discriminator
        #self.detector.convert_bn_to_dial(self.detector, device=next(kp_detector.parameters()).device)
        self.heatmap_res = self.detector.heatmap_res
        self.geo_transform = geo_transform
        self.train_params = train_params
        self.to_skeleton = kp_to_skl
        
        
        self.epoch_ratio = 0
        self.angle_range = 180 
        self.angle_incr_factor = 15
        self.angle_loss = torch.nn.MSELoss()

        self.percentile = None

    def equivariance_loss(self, source, transformed, geo_param):
        forward = l1_loss(self.geo_transform(source, geo_param), 
                           transformed).mean()
        return forward 
    

    def make_coordinate_grid(self, spatial_size, type):
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

    def gaussian2kp(self, heatmap, clip_variance=None):
        """
        Extract the mean and the variance from a heatmap
        """
        shape = heatmap.shape
        kp = {}
        kp['var'] = torch.var(heatmap,(2,3))
        heatmap = heatmap.unsqueeze(-1) + 1e-7
        grid_ = self.make_coordinate_grid(shape[2:], heatmap.type())
        grid = grid_.unsqueeze(0).unsqueeze(0)
        grid =  grid.to(heatmap.device)
        mean_ = (heatmap * grid)
        mean = mean_.sum(dim=(2, 3))

        kp['mean'] =  mean 

        return kp    

    def generate_labels(self, label_generator, images, annots, device="cuda",kp_map=None, heatmap_size=(122,122), heatmap_var=0.15, semi_supervised=False, reverse=False):

        with torch.no_grad():
            out = label_generator(images)
            hm  = out["heatmaps"]
        if not reverse:
            kps = self.gaussian2kp(hm[:,kp_map].detach())["mean"]
        else:
            kps = self.gaussian2kp(hm.detach())["mean"]

        with torch.no_grad():
            maps = self.discriminator(hm)  

        probabilities = mean_batch(maps[0]).cpu().numpy()
        index_sorted = np.where(probabilities<self.percentile)[0]

        # if semisupervised return the annotation of the one selected
        if semi_supervised:
            return images[index_sorted], annots[index_sorted],  index_sorted
        else:
            return images[index_sorted], kps[index_sorted],  index_sorted

    def adapt(self, images, annots, label_generator=None, kp_map=None,  treshold=True, affine=False, reverse=False, device="cuda"):
        kps, heatmaps, tgt_images, kps_pseud, reg_loss, pseudo_loss= None, None, None, None, 0,0
        ground_truth = None
        images_selected = None
    
        dict_out = self.detector(images)
        if not reverse:
            kps = dict_out['value'][:, kp_map]
            kps = unnorm_kp(kps)
            heatmaps = dict_out['heatmaps'][:, kp_map]
        else:
            kps = dict_out['value']
            kps = unnorm_kp(kps)
            heatmaps = dict_out['heatmaps']

        geo_loss = 0
        geo_dict_out = None
        geo_images = None
        if self.geo_transform is not None:
            geo_loss = 0
            for i in range(self.train_params["num_augmentation"]):
                range_angle = min(self.angle_range ,int(self.angle_incr_factor * (self.angle_range* self.epoch_ratio)))
                angle = random.randint(-1*range_angle,range_angle)
                geo_images = self.geo_transform(images, angle).detach()
                geo_dict_out = self.detector(geo_images)
                if not reverse:
                    kps_rot =  geo_dict_out["value"][:, kp_map]
                    kps_rot = unnorm_kp(kps_rot)
                    geo_heatmaps = geo_dict_out['heatmaps'][:, kp_map]
                else:
                    kps_rot = geo_dict_out["value"]
                    kps_rot = unnorm_kp(kps_rot)
                    geo_heatmaps = geo_dict_out['heatmaps']

                geo_loss += self.equivariance_loss(heatmaps, geo_heatmaps, angle)

        if self.train_params["regularize"]:

            dict_out_source = label_generator(images)
            if not reverse:
                heatmaps_source = dict_out_source['heatmaps'][:, kp_map]
            else:
                heatmaps_source = dict_out_source['heatmaps']

            reg_loss = 0 
            reg_loss = l1_loss(heatmaps,heatmaps_source).mean()

        if self.train_params["use_discriminator"]:
            images_selected, gt, index_sorted  = self.generate_labels(label_generator, images, annots, kp_map=kp_map, device=device,  semi_supervised=self.train_params["semi_supervise"], reverse=reverse)

            if gt.shape[0]>0:
                if self.train_params["semi_supervise"] and reverse:
                    gt = gt[:,kp_map]
                size = (self.heatmap_res, self.heatmap_res)
                variance = self.train_params['heatmap_var']
                heatmaps_source = kp2gaussian2(gt, size, variance).detach()
                pseudo_loss = l1_loss(heatmaps[index_sorted],heatmaps_source).mean()
            ground_truth =  unnorm_kp(gt).cpu()


        return {"keypoints": kps,
                "heatmaps": heatmaps,
                "geo_loss": geo_loss,
                "reg_loss" : reg_loss,
                "pseudo_loss": pseudo_loss,
                "geo_out": geo_dict_out,
                "geo_images": geo_images,
                "kps_pseud": ground_truth,
                "image_pseudo": images_selected,
                }

def train_kpdetector(model_kp_detector,
                       label_generator,
                       loaders,
                       train_params,
                       checkpoint,
                       logger, device_ids, model_discriminator=None,
                       kp_map=None,reverse=False):
    log_params = train_params['log_params']
    resume_epoch = 0
    resume_iteration = 0
    optimizer_kp_detector = torch.optim.Adam(model_kp_detector.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, 
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
                                    geo_transform=geo_transform)

    print(f"regularize {train_params['regularize']}")
    print(f"use_discriminator {train_params['use_discriminator']}")
    print(f"regularize {train_params['regularize']}")

    k = 0
    if train_params['test'] == True:
        results = evaluate(model_kp_detector, loader_tgt_test, dset=train_params['dataset'])
        print(' MSE: ' + str(results['MSE']) + ' PCK: ' + str(results['PCK'])) 
        return

    heatmap_var = train_params['heatmap_var']
    heatmap_size = (kp_detector.heatmap_res, kp_detector.heatmap_res)

    loader_tgt_train, loader_tgt_test = loaders
    device_ids =device_ids[0]

    if train_params['use_discriminator']:
        disc_percentile =  discriminator_percentile(model_kp_detector, loader_tgt_train, model_discriminator,  device=device_ids, percentile=train_params["percentile"])
        kp_detector.percentile = disc_percentile
        print(f"discriminator percentile settet at : {kp_detector.percentile}")

    for epoch in range(logger.epoch, train_params['num_epochs']):

        kp_detector.epoch_ratio = epoch/train_params["num_epochs"]
        #kp_detector.detector.set_domain_all(source=False)
        results_tgt_test = evaluate(kp_detector.detector, loader_tgt_test, dset=train_params['tgt_train'], filter=kp_map, device=device_ids, reverse=reverse)
        results_tgt_train = evaluate(kp_detector.detector, loader_tgt_train, dset=train_params['tgt_train'], filter=kp_map, device=device_ids, reverse=reverse)

        print('Epoch ' + str(epoch)+ ' PCK_target_test: ' + str(results_tgt_test['PCK']))
        print('Epoch ' + str(epoch)+ ' PCK_target_train: ' + str(results_tgt_train['PCK']))

        logger.add_scalar('MSE target test', results_tgt_test['MSE'], epoch)
        logger.add_scalar('PCK target test', results_tgt_test['PCK'], epoch)

        logger.add_scalar('MSE target train', results_tgt_train['MSE'], epoch)
        logger.add_scalar('PCK target train', results_tgt_train['PCK'], epoch)
   
        for i, tgt_batch  in enumerate(tqdm(loader_tgt_train)):
            tgt_images = tgt_batch['imgs'].to(device_ids)
            tgt_annots = tgt_batch['annots'].to(device_ids)
    
            kp_detector_tgt_out = kp_detector.adapt(tgt_images, tgt_annots,label_generator=label_generator, kp_map=kp_map, device=device_ids, reverse=reverse)
            
            geo_loss_value = 0
            geo_loss = 0
            if train_params['use_rotation']:
                geo_out = kp_detector_tgt_out['geo_out']
                geo_loss = train_params['loss_weights']['geometric'] * \
                    (kp_detector_tgt_out['geo_loss'])
                geo_loss_value = geo_loss.item()
            
            pseudo_loss = 0
            if train_params['use_discriminator']:
                pseudo_loss_item = 0
                pseudo_loss = train_params['loss_weights']['pseudo_loss'] * kp_detector_tgt_out['pseudo_loss']
                if pseudo_loss != 0:
                    pseudo_loss_item = pseudo_loss.item() 
            
            reg_loss = 0
            if train_params["regularize"]:
                reg_loss = train_params['loss_weights']['source_regression'] * kp_detector_tgt_out['reg_loss']
                loss =  geo_loss + reg_loss +  pseudo_loss
                loss.backward()

            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()

            ####### LOG

            if train_params['use_rotation']:
                logger.add_scalar('tgt geo loss',
                                geo_loss_value,
                                logger.iterations)

            if train_params['regularize']:
                logger.add_scalar('reg loss',
                                reg_loss,
                                logger.iterations)

            if train_params['use_discriminator']:
                logger.add_scalar('pseudo loss', 
                        pseudo_loss_item, 
                        logger.iterations)
            

            if i in log_params['log_imgs']:
                if kp_detector_tgt_out['image_pseudo'] is not None:
                    if kp_detector_tgt_out['image_pseudo'].shape[0]>0:
                        concat_img_tgt_pseudo = np.concatenate((
                            draw_kp(tensor_to_image(tgt_images[k]), 
                                    unnorm_kp(tgt_annots[k])),
                            draw_kp(tensor_to_image(kp_detector_tgt_out["image_pseudo"][k]), 
                                    kp_detector_tgt_out['kps_pseud'][k], color='red')),
                                    axis=2)
                        logger.add_image('pseudo labels', concat_img_tgt_pseudo, logger.iterations)
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

                heatmap_0 = kp_detector_tgt_out['heatmaps'][k, 0].unsqueeze(0)
                heatmap_1 = kp_detector_tgt_out['heatmaps'][k, 1].unsqueeze(0)
                concat_hm_tgt_net = np.concatenate((
                    tensor_to_image(heatmap_0, True),
                    tensor_to_image(heatmap_1, True)), axis= 2)
                
                logger.add_image('tgt train', concat_img_tgt, logger.iterations)
                logger.add_image('tgt_heatmap', concat_hm_tgt_net, logger.iterations)

            logger.step_it()

        scheduler_kp_detector.step()
        logger.step_epoch(models = {'model_kp_detector':model_kp_detector,
                                    'optimizer_kp_detector':optimizer_kp_detector})

def draw_kp(img_, kps, color='blue'):
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


