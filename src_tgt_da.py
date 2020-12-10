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

### compute the angle difference between the kps
def angle_difference(kps,rot_kps,gt_angle,loss, device="cuda"):
    angle = torch.atan2(torch.det(torch.stack([rot_kps,kps], 2)),torch.sum(rot_kps*kps, dim=2))
    gt_angle = torch.tensor(gt_angle).type(torch.FloatTensor)
    gt_angle = gt_angle.to(device)
    angle = (angle*180/np.pi).mean()
    return loss(angle,gt_angle)


class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector, train_params, 
                       discriminator=None, 
                       geo_transform=None,
                       kp_to_skl=None, angle_difference=False):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.detector.convert_bn_to_dial(self.detector, device=next(kp_detector.parameters()).device)
        self.discriminator = discriminator
        self.heatmap_res = self.detector.heatmap_res
        self.geo_transform = geo_transform
        self.train_params = train_params
        self.loss_weights = self.train_params['loss_weights']
        self.to_skeleton = kp_to_skl
        
        self.epoch_ratio = 0
        self.angle_range = 180 #45
        self.angle_incr_factor = 15
        self.color_jitter = ColorJitter(brightness = 0,contrast=0,saturation=0,hue=0.5)
        self.to_pil_image = ToPILImage()
        self.use_angle_equi = angle_difference
        self.angle_loss = torch.nn.MSELoss()
        #self.percentile = 2.6430241160113612e-08 #6.463432065118013e-08 # 1%
        #self.percentile = 3.696300758804228e-08 #7.263018737546644e-08 # 5%
        self.percentile = 0.06445252522826195


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
            angle = 90
            geo_images = self.geo_transform(images, angle).detach()
            geo_dict_out = self.detector(geo_images)
            geo_heatmaps = geo_dict_out['heatmaps'] if kp_map is None else geo_dict_out['heatmaps'][:, kp_map]
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
        # backward = l1_loss(source,
        #                     self.geo_transform(transformed, -geo_param)).mean()
        return forward #+ backward
    

    def geo_consistency_affine(self, y, y_affine,aff_matrix):
        #out = {}
        # [batchsize, num_kp, 122, 122]
        # apply the affine transf to the non rotated heatmpas
        orig_to_aff_hm = batch_affine(y,aff_matrix, inverse=False)
        loss = l1_loss(orig_to_aff_hm, y_affine).mean()

        # # apply the inverse affine trans to the rotated heatmaps
        # inverse_aff = inverse_aff_values(aff_matrix)
        # aff_to_orig_hm = batch_affine(y_affine,inverse_aff, inverse=True)

        # out['t_inv'] = l1_loss(aff_to_orig_hm, y).mean()
        return loss 

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
        #adding small eps to avoid 'nan' in variance
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

    def generate_labels(self, label_generator, images, annots,  num_augmentation=5, treshold=True ,use_heatmaps=True,mask=None, device="cuda", range_angle=10,increase_sample_f=4 ,kp_map=None, all_kp=False,heatmap_size=(122,122), heatmap_var=0.15):
        heatmaps = torch.Tensor([]).to(device)
        kp_lists = torch.Tensor([]).to(device)
        with torch.no_grad():
            for n_aug in range(num_augmentation):
                #print(f"n_aug {n_aug}")
                angle = random.randint(-1*range_angle,range_angle)
                rot_img = batch_image_rotation(images, angle)
                rot_img = rot_img.to(device)
                out = label_generator(rot_img)
                if kp_map is not None:
                    heatmaps = torch.cat((heatmaps, batch_image_rotation(out["heatmaps"], -angle).unsqueeze(1)), axis=1)
                    heatmaps = heatmaps.sum(axis=1).unsqueeze(1) #torch.cat((heatmaps, batch_image_rotation(out["heatmaps"], -angle).unsqueeze(1)), axis=1)
                else:
                    heatmaps = torch.cat((heatmaps, batch_image_rotation(out["heatmaps"], -angle).unsqueeze(1)), axis=1) 
                    heatmaps = heatmaps.sum(axis=1).unsqueeze(1)

        hm  = heatmaps.sum(axis=1)/num_augmentation #heatmaps.sum(axis=1)/heatmaps.shape[1]
        
        if all_kp:
            #kps = self.gaussian2kp(hm[:,kp_map].detach())["mean"]
            #hm = kp2gaussian2(kps, heatmap_size, heatmap_var).detach()
            return None, hm, None

        if treshold:
            gt_heatmaps = kp2gaussian2(annots.to(device), heatmap_size,heatmap_var).detach()
            tgt_loss = masked_l2_heatmap_loss(hm[:,kp_map], 
                    gt_heatmaps, mask).cpu().numpy()
            index_sorted = np.where(tgt_loss<self.percentile)[0]
            return images[index_sorted], out["value"][:,kp_map][index_sorted], mask[index_sorted], index_sorted
        else:
            # heatmaps shape [batch, num_augmentaiton, nkp, w_hm, h_hm]
            kps = self.gaussian2kp(hm[:,kp_map].detach())["mean"]
            hm = kp2gaussian2(kps, heatmap_size, heatmap_var).detach()
            generator_scores = [] 
            with torch.no_grad():
                maps = self.discriminator(hm)  
            probabilities = mean_batch(maps[0]).cpu().numpy()
            index_sorted = np.where(probabilities<self.percentile)[0]
            #print(len(index_sorted))
            return images[index_sorted], kps[index_sorted], mask[index_sorted], index_sorted

    def adapt(self, images, annots,jit_imgs=None, label_generator=None, kp_map=None, mask=None, treshold=True,device="cuda", regularize=False, jitter=False, affine=False, reverse=False,repeat=2):
        kps, heatmaps, tgt_loss, geo_loss, geo_out, geo_images, tgt_images, kps_pseud, generator_loss, reg_loss, jit_dict_out, jit_images= None, None, 0, 0, None, None, None, None, 0, 0, None, None 
        ground_truth = None
        jit_loss = 0
        self.detector.set_domain_all(source=False)
        images_selected = torch.Tensor([0])
        if mask is not None:
            mask = mask.to(images.device)
        if label_generator is not None and not regularize:
            images_selected, ground_truth, mask = self.generate_labels(label_generator, images, annots,treshold=treshold, kp_map=kp_map, device=device, mask=mask)     

        if images_selected.shape[0] > 0 and label_generator is not None and not regularize:
            #print(f"images_selected {images_selected.shape}")
            gt = ground_truth
            tgt_loss = 0
            dict_out = self.detector(images_selected)
            if not reverse:
                kps = dict_out['value'][:, kp_map]
                kps = unnorm_kp(kps)
                heatmaps = dict_out['heatmaps'][:, kp_map] if kp_map is not None else dict_out['heatmaps']
            else:
                kps = dict_out['value']
                kps = unnorm_kp(kps)
                heatmaps = dict_out['heatmaps']

            if self.train_params['source_loss'] == 'kp_values':
                tgt_loss = masked_l2_loss(dict_out['value'], gt, mask)
            elif self.train_params['source_loss'] == 'heatmap':
                size = (self.heatmap_res, self.heatmap_res)
                variance = self.train_params['heatmap_var']
                gt_heatmaps = kp2gaussian2(gt, size, variance).detach()
                tgt_loss = masked_l2_heatmap_loss(heatmaps, 
                                                gt_heatmaps, mask).mean()
            geo_loss = 0
            geo_dict_out = None
            geo_images = None
            generator_loss = 0
            ground_truth =  unnorm_kp(ground_truth).cpu()
        else:
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
                for i in range(repeat):
                    if not affine:
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
                        if self.use_angle_equi:
                            geo_loss += angle_difference(kps, kps_rot, angle, self.angle_loss, device=device)
                        else:
                            geo_loss += self.equivariance_loss(heatmaps, geo_heatmaps, angle)
                    else:
                        geo_images, aff_matrix = batch_img_affine(images) # uses the predefined angle translation and ranges
                        geo_images = geo_images.detach()
                        geo_dict_out = self.detector(geo_images.to(device))
                        if not reverse:
                            geo_heatmaps = geo_dict_out['heatmaps'][:, kp_map]
                        else:
                            geo_heatmaps = geo_dict_out['heatmaps']
                        geo_loss += self.geo_consistency_affine(heatmaps, geo_heatmaps, aff_matrix)

            if jitter:
                range_angle = min(self.angle_range ,int(self.angle_incr_factor * (self.angle_range* self.epoch_ratio)))
                angle = random.randint(-1*range_angle,range_angle)
                jit_images = self.geo_transform(jit_imgs, angle).detach()
                jit_dict_out = self.detector(jit_images)
                if not reverse:
                    jit_heatmaps = jit_dict_out['heatmaps'][:, kp_map]
                else:
                    jit_heatmaps = jit_dict_out['heatmaps']

                jit_loss = self.equivariance_loss(heatmaps, jit_heatmaps, angle)
            if regularize:
                dict_out_source = label_generator(images)
                if not reverse:
                    heatmaps_source = dict_out_source['heatmaps'][:, kp_map]
                else:
                    heatmaps_source = dict_out_source['heatmaps']
                reg_loss = 0 
                reg_loss = l1_loss(heatmaps,heatmaps_source).mean()
                images_selected, gt, _, index_sorted  = self.generate_labels(label_generator, images, annots, kp_map=kp_map, device=device, mask=mask, all_kp=False, treshold=False, num_augmentation=1, range_angle=0)
                #print(gt.shape)
                #print(gt)
                pseudo_loss = 0
                if gt.shape[0]>0:
                    size = (self.heatmap_res, self.heatmap_res)
                    variance = self.train_params['heatmap_var']
                    heatmaps_source = kp2gaussian2(gt, size, variance).detach()
                    #print(heatmaps_source.shape)
                    pseudo_loss = l1_loss(heatmaps[index_sorted],heatmaps_source).mean() #self.equivariance_loss(heatmaps, heatmaps_soruce, 0)
                ground_truth =  unnorm_kp(gt).cpu()

            generator_loss = 0

        return {"keypoints": kps,
                "heatmaps": heatmaps,
                "tgt_loss" : tgt_loss,
                "geo_loss": geo_loss,
                "jit_loss": jit_loss,
                "reg_loss" : reg_loss,
                "pseudo_loss": pseudo_loss,
                "geo_out": geo_dict_out,
                "geo_images": geo_images,
                "jit_out": jit_dict_out,
                "jit_images": jit_images,
                "kps_pseud": ground_truth,
                "image_pseudo": images_selected,
                "generator_loss": generator_loss,
                }

def train_kpdetector(model_kp_detector,
                       label_generator,
                       loaders,
                       train_params,
                       checkpoint,
                       logger, device_ids, model_discriminator=None,
                       kp_map=None, adapt_pretrained= False, reverse=False):
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
                                    geo_transform=geo_transform, angle_difference=train_params["angle_equivariance"])

    # select the labels on a treshold base
    treshold = True
    if train_params['use_gan']:
        treshold = False
    affine_augmentation = train_params["affine_augmentation"]
    print(f"use_treshold {treshold}")
    print(f"affine_augmentation {affine_augmentation}")
    print(f"regularize {train_params['regularize']}")
    print(f"jitter {train_params['jitter']}")
    print(f"angle_equi {train_params['angle_equivariance']}")

    k = 0
    if train_params['test'] == True:
        results = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'])
        print(' MSE: ' + str(results['MSE']) + ' PCK: ' + str(results['PCK'])) 
        return

    # number of epoch after those epochs we stop the discriminator training and we use the disc value as classification
    train_disc_only = 0

    heatmap_var = train_params['heatmap_var']
    heatmap_size = (kp_detector.heatmap_res, kp_detector.heatmap_res)
    loader_src_train, loader_src_test, loader_tgt = loaders
    iterator_source = iter(loader_src_train)
    device_ids =device_ids[0]
    for epoch in range(logger.epoch, train_params['num_epochs']):

        kp_detector.epoch_ratio = epoch/train_params["num_epochs"]
        kp_detector.detector.set_domain_all(source=False)
        results_tgt = evaluate(kp_detector.detector, loader_tgt, dset=train_params['tgt_train'], filter=kp_map, device=device_ids, reverse=reverse)
        
        if adapt_pretrained == False:
            kp_detector.detector.set_domain_all(source=True)
            results_src_test = evaluate(kp_detector.detector, loader_src_test, dset=train_params['src_test']) 
            results_src_train = evaluate(kp_detector.detector, loader_src_train, dset=train_params['src_train'])

        print('Epoch ' + str(epoch)+ ' PCK_target: ' + str(results_tgt['PCK']))
        logger.add_scalar('MSE target', results_tgt['MSE'], epoch)
        logger.add_scalar('PCK target', results_tgt['PCK'], epoch)

        # ## here evaluation of the discriminator
        # if train_params['use_gan'] and epoch == 10:
        #     logger.save_model(models = {'model_kp_discriminator':model_discriminator,
        #                 'optimizer_kp_discriminator':optimizer_discriminator}) 
        #     print(f"saved discriminator")            
        # if adapt_pretrained == False or (train_params['use_gan'] and epoch == train_disc_only):
        #     print(f"Finding the treshold.....")        
        #     disc_percentile =  discriminator_percentile(model_kp_detector, loader_tgt, model_discriminator,  device=device_ids,filter=kp_map, percentile=95) ## ignore from reverse
        #     kp_detector.percentile = disc_percentile
        #     ### save the discriminator trained 
        #     logger.save_model(models = {'model_kp_discriminator':model_discriminator,
        #                             'optimizer_kp_discriminator':optimizer_discriminator}) 
        #     print(f"Unplugged discriminator {disc_percentile} training only detector") 
        #     break

        if adapt_pretrained == False:
            logger.add_scalar('MSE src train', results_src_train['MSE'], epoch)
            logger.add_scalar('PCK src train', results_src_train['PCK'], epoch)
            logger.add_scalar('MSE src test', results_src_test['MSE'], epoch)
            logger.add_scalar('PCK src test', results_src_test['PCK'], epoch)
   
        for i, tgt_batch  in enumerate(tqdm(loader_tgt)):
            tgt_images = tgt_batch['imgs'].to(device_ids)
            tgt_annots = tgt_batch['annots'].to(device_ids)
            if train_params["jitter"]:
                jit_image = tgt_batch['jit_imgs'].to(device_ids)
            else:
                jit_image =  None
            mask_tgt = None if 'kp_mask' not in tgt_batch.keys() else tgt_batch['kp_mask']

            if adapt_pretrained == False or (train_params['use_gan'] and epoch < train_disc_only):
                try:
                    src_batch = next(iterator_source)
                except:
                    iterator_source = iter(loader_src_train)
                    src_batch = next(iterator_source)
                src_images = src_batch['imgs'].to(device_ids)
                src_annots = src_batch['annots'].to(device_ids)
                src_annots_hm = kp2gaussian2(src_annots, heatmap_size, heatmap_var)

                mask = None if 'kp_mask' not in src_batch.keys() else src_batch['kp_mask']
            
            
            if adapt_pretrained == False:
                kp_detector_src_out = kp_detector(src_images, src_annots, mask)

    
            if not train_params['use_gan'] or epoch >= train_disc_only:
                kp_detector_tgt_out = kp_detector.adapt(tgt_images, tgt_annots,jit_imgs=jit_image,label_generator=label_generator, kp_map=kp_map, mask=mask_tgt, device=device_ids, treshold=treshold,regularize=train_params["regularize"], jitter=train_params["jitter"], affine=affine_augmentation, reverse=reverse)

                if kp_detector_tgt_out['tgt_loss'] != 0:
                    tgt_loss = kp_detector_tgt_out['tgt_loss'].mean()
                else:
                    tgt_loss = 0 

                if adapt_pretrained == False:
                    src_loss = kp_detector_src_out['src_loss'].mean() 
                
                geo_loss_value = 0
                geo_loss = 0
                if train_params['use_rotation'] and adapt_pretrained == False:
                    geo_out = kp_detector_tgt_out['geo_out']
                    geo_loss = train_params['loss_weights']['geometric'] * \
                        (kp_detector_tgt_out['geo_loss'] + kp_detector_src_out['geo_loss'])
                    geo_loss_value = geo_loss.item()
                elif train_params['use_rotation']:
                    geo_out = kp_detector_tgt_out['geo_out']
                    geo_loss = train_params['loss_weights']['geometric'] * \
                        (kp_detector_tgt_out['geo_loss'])
                    geo_loss_value = geo_loss.item()
                jit_loss = 0
                if train_params["jitter"]:
                    jit_loss = train_params['loss_weights']['jitter'] * \
                        (kp_detector_tgt_out['jit_loss'])
                    jit_loss_value = jit_loss.item()

                gen_loss = 0
                gen_loss_value = 0
                pseudo_loss = 0
                if train_params['use_gan']:
                    gen_loss = kp_detector_tgt_out['generator_loss']
                    pseudo_loss = train_params['loss_weights']['pseudo_loss'] * kp_detector_tgt_out['pseudo_loss']
                    gen_loss_value = 0 #gen_loss.item()
                    pseudo_loss_item = pseudo_loss.item() 
                    
                if label_generator is not None and not train_params["regularize"]:
                    if tgt_loss != 0:
                        loss = geo_loss + gen_loss + tgt_loss
                    else:
                        loss = geo_loss + gen_loss
                    if loss != 0:
                        loss.backward()
                
                reg_loss = 0
                if train_params["regularize"]:
                    reg_loss = train_params['loss_weights']['source_regression'] * kp_detector_tgt_out['reg_loss']
                    loss =  geo_loss + reg_loss + jit_loss + pseudo_loss
                    loss.backward()

                if adapt_pretrained == False:
                    loss = geo_loss + src_loss + gen_loss
                    loss.backward()
                # if train_params["use_gan"]:
                #     loss = geo_loss + gen_loss
                #     loss.backward()

                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
            else:
                kp_detector_tgt_out = kp_detector.adapt(tgt_images, tgt_annots,label_generator=None, kp_map=kp_map, mask=mask_tgt, device=device_ids, treshold=treshold,regularize=train_params["regularize"])

            disc_loss_value = 0

            if train_params['use_gan'] and epoch < train_disc_only:
                hm = kp_detector_tgt_out['heatmaps'].detach()
                kp = kp_detector.gaussian2kp(hm)["mean"]
                hm = kp2gaussian2(kp, heatmap_size, heatmap_var).detach()
                if not reverse:
                    disc_out = discriminator(src_annots_hm[:, kp_map].detach(), 
                                            hm, filter=kp_map)
                else:
                    disc_out = discriminator(src_annots_hm.detach(), 
                                            hm, filter=kp_map)
                disc_loss = disc_out['loss'].mean()
                disc_loss_value = disc_loss.item()
                optimizer_discriminator.zero_grad()
                disc_loss.backward()
                optimizer_discriminator.step()
                optimizer_kp_detector.zero_grad()




            ####### LOG
            if adapt_pretrained == False:
                logger.add_scalar('src l2 loss', 
                                    src_loss.item(), 
                                    logger.iterations)

            if not train_params['use_gan'] or epoch >= train_disc_only:

                if tgt_loss!= 0 :
                    tgt_loss =  tgt_loss.item()


                logger.add_scalar('sum loss', 
                        loss, 
                        logger.iterations)
                

                logger.add_scalar('tgt l2 loss', 
                        tgt_loss, 
                        logger.iterations)
                    
                logger.add_scalar('tgt geo loss',
                                geo_loss_value,
                                logger.iterations)

                logger.add_scalar('reg loss',
                                reg_loss,
                                logger.iterations)
                logger.add_scalar('gen loss',
                               gen_loss_value,
                               logger.iterations)
                if train_params["jitter"]:
                    logger.add_scalar('jit loss',
                                jit_loss_value,
                                logger.iterations)
            if train_params['use_gan']:
                logger.add_scalar('pseudo loss', 
                        pseudo_loss_item, 
                        logger.iterations)

            logger.add_scalar('disc loss',
                               disc_loss_value,
                               logger.iterations)
            
            

            if i in log_params['log_imgs']:
                if not train_params['use_gan'] or epoch >= train_disc_only:
                    if kp_detector_tgt_out['image_pseudo'].shape[0]>1:
                        #print(kp_detector_tgt_out['image_pseudo'].shape)
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
                    if train_params['jitter']:
                        concat_img_jitter = np.concatenate((
                            draw_kp(tensor_to_image(tgt_images[k]), 
                                    kp_detector_tgt_out['keypoints'][k], color='red'),
                            draw_kp(tensor_to_image(kp_detector_tgt_out['jit_images'][k]), 
                                    unnorm_kp(kp_detector_tgt_out['jit_out']['value'][k]), color='red')),
                                    axis=2)

                        logger.add_image('jit tgt', concat_img_jitter, logger.iterations)


                    heatmap_0 = kp_detector_tgt_out['heatmaps'][k, 0].unsqueeze(0)
                    heatmap_1 = kp_detector_tgt_out['heatmaps'][k, 1].unsqueeze(0)
                    concat_hm_tgt_net = np.concatenate((
                        tensor_to_image(heatmap_0, True),
                        tensor_to_image(heatmap_1, True)), axis= 2)
                    if  adapt_pretrained == False:
                        heatmap_0 = src_annots_hm[k, 0].unsqueeze(0)
                        heatmap_1 = src_annots_hm[k, 1].unsqueeze(0)
                        concat_hm_gt = np.concatenate((
                            tensor_to_image(heatmap_0, True),
                            tensor_to_image(heatmap_1, True)), axis= 2)
                    
                    logger.add_image('tgt train', concat_img_tgt, logger.iterations)
                    logger.add_image('tgt_heatmap', concat_hm_tgt_net, logger.iterations)
                    if  adapt_pretrained == False:
                        logger.add_image('src_heatmap', concat_hm_src_net, logger.iterations)
                        logger.add_image('gt heatmap', concat_hm_gt, logger.iterations)
                #k += 1
                #k = k % len(log_params['log_imgs']) 
            logger.step_it()
        if epoch >= train_disc_only:
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


