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
from modules.util import kp2gaussian2, gaussian2kp
from modules.util import batch_image_rotation, batch_kp_rotation 
from datasets.annot_converter import HUMANS_TO_HUMANS
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw

class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector, geo_transform=None, angle_equivariance=False):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.detector.convert_bn_to_dial(self.detector)
        self.heatmap_res = self.detector.heatmap_res
        self.geo_transform = geo_transform
        self.angle_equivariance = angle_equivariance
        self.geo_f = torch.nn.MSELoss()


    def forward(self, images, ground_truth, mask=None, kp_map=None):
        self.detector.set_domain_all(source=True)
        dict_out = self.detector(images)
        gt = ground_truth if kp_map is None else ground_truth[:, kp_map]
        loss = masked_l2_loss(dict_out['value'], gt, mask)
        #loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)
        heatmaps = dict_out['heatmaps'][:, kp_map] if kp_map is not None else dict_out['heatmaps']
        geo_loss = 0
        if self.geo_transform is not None:
            angle = 90
            geo_images = self.geo_transform(images, angle)
            geo_dict_out = self.detector(geo_images)
            geo_heatmaps = geo_dict_out['heatmaps'] if kp_map is None else geo_dict_out['heatmaps'][:, kp_map]
            if self.angle_equivariance:
                geo_loss = self.angle_difference(dict_out['value'],geo_dict_out['value'], angle)
            else:   
                geo_loss = self.equivariance_loss(heatmaps, geo_heatmaps, angle)

        kps = unnorm_kp(dict_out['value'])
        return {"keypoints": kps,
                "heatmaps": dict_out['heatmaps'],
                "l2_loss": loss.mean(),
                "geo_loss": geo_loss,
                }
    def equivariance_loss(self, source, transformed, geo_param):
        forward = l1_loss(self.geo_transform(source, geo_param), 
                           transformed).mean()
        backward = l1_loss(source,
                            self.geo_transform(transformed, -geo_param)).mean()
        return forward + backward
    
    def angle_difference(self,kps,rot_kps,gt_angle, device="cuda"):
    
       # v0 =  -1* rot_kps
       # v1 = -1*  kps

        # stack = torch.stack([rot_kps,kps], 2)
        # det =  torch.det(stack)
        # dot = torch.sum(rot_kps*kps, dim=2)

        angle = torch.atan2(torch.det(torch.stack([rot_kps,kps], 2)),torch.sum(rot_kps*kps, dim=2))
        gt_angle = torch.tensor(gt_angle).type(torch.FloatTensor)
        gt_angle = gt_angle.cuda()
        angle = (angle*180/np.pi).mean()
        return self.geo_f(angle,gt_angle)

    def adapt(self, images, kp_map):
        self.detector.set_domain_all(source=False)
        dict_out = self.detector(images)
        kps = dict_out['value'][:, kp_map]
        heatmaps = dict_out['heatmaps'][:, kp_map]
        geo_loss = 0
        if self.geo_transform is not None:
            angle = 90
            geo_images = self.geo_transform(images, angle)
            geo_dict_out = self.detector(geo_images)
            geo_heatmaps = geo_dict_out['heatmaps'][:, kp_map]
            if self.angle_equivariance:
                geo_loss = self.angle_difference(kps,geo_dict_out['value'][:,kp_map], angle)
            else:    
                geo_loss = self.equivariance_loss(heatmaps, geo_heatmaps, angle)
        kps = unnorm_kp(kps)

        return {"keypoints": kps,
                "heatmaps": heatmaps,
                "geo_loss": geo_loss,
                "geo_out": geo_dict_out,
                "geo_images": geo_images,
                }



def train_kpdetector(model_kp_detector,
                       loaders,
                       train_params,
                       checkpoint,
                       logger, device_ids, kp_map=None):
    log_params = train_params['log_params']
    optimizer_kp_detector = torch.optim.Adam(model_kp_detector.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    resume_epoch = 0
    resume_iteration = 0
    if checkpoint is not None:
        print('Loading Checkpoint: %s' % checkpoint)
        if train_params['test'] == False:
            resume_epoch, resume_iteration = logger.checkpoint.load_checkpoint(checkpoint,
                                                  model_kp_detector=model_kp_detector,
                                                  optimizer_kp_detector=optimizer_kp_detector)
        logger.epoch = resume_epoch
        logger.iterations = resume_iteration

    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)
    kp_detector = KPDetectorTrainer(model_kp_detector, batch_image_rotation, angle_equivariance=train_params['angle_equivariance'])
    #kp_detector = DataParallelWithCallback(kp_detector, device_ids=device_ids)
    k = 0
    if train_params['test'] == True:
        results = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'])
        print(' MSE: ' + str(results['MSE']) + ' PCK: ' + str(results['PCK'])) 
        return

    heatmap_var = train_params['heatmap_var']

    loader_src_train, loader_src_test, loader_tgt = loaders
    iterator_source = iter(loader_src_train)
    for epoch in range(logger.epoch, train_params['num_epochs']):
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

            mask = None if 'kp_mask' not in src_batch.keys() else src_batch['kp_mask']
            kp_detector_src_out = kp_detector(src_images, src_annots, mask)
            kp_detector_tgt_out = kp_detector.adapt(tgt_images, kp_map)
            geo_out = kp_detector_tgt_out['geo_out']

            geo_loss = train_params['loss_weights']['geometric'] * \
                       (kp_detector_tgt_out['geo_loss'] + kp_detector_src_out['geo_loss'])
            src_loss = kp_detector_src_out['l2_loss'].mean() 
            if epoch < 3:
                loss = src_loss
                loss.backward()
            else:              
                loss = geo_loss + src_loss
                loss.backward() 


            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()

            ####### LOG
            logger.add_scalar('src l2 loss', 
                               src_loss.item(), 
                               logger.iterations)
            logger.add_scalar('tgt geo loss',
                               geo_loss.item(),
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
                concat_img_geo = np.concatenate((
                    draw_kp(tensor_to_image(tgt_images[k]), 
                             kp_detector_tgt_out['keypoints'][k], color='red'),
                    draw_kp(tensor_to_image(kp_detector_tgt_out['geo_images'][k]), 
                             unnorm_kp(kp_detector_tgt_out['geo_out']['value'][k]), color='red')),
                             axis=2)

                logger.add_image('src train', concat_img_src, logger.iterations)
                logger.add_image('tgt train', concat_img_tgt, logger.iterations)
                logger.add_image('geo tgt', concat_img_geo, logger.iterations)
                k += 1
                k = k % len(log_params['log_imgs']) 
            logger.step_it()

        scheduler_kp_detector.step()
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


