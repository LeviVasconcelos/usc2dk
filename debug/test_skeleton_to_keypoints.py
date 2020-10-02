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
from modules.losses import l2_loss, discriminator_loss, generator_loss_2d, l1_loss
from torch.optim.lr_scheduler import MultiStepLR
from modules.vgg19 import Vgg19
from modules.util import batch_kp_rotation, batch_image_rotation
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw

def flatten_views(img):
    return img.view(-1, *img.shape[2:])

def tensor_to_image(x):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    out = (out * 255).astype(np.uint8)
    return out

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


class SkeletonToKeypoints(nn.Module):
    def __init__(self, model_skeleton_to_keypoint,
                 model_keypoint_to_skeleton):
        super(SkeletonToKeypoints, self).__init__()
        self.encoder = model_skeleton_to_keypoint
        self.decoder = model_keypoint_to_skeleton

    def forward(self, skeleton, gt_kps):
        keypoints = self.encoder(skeleton)
        loss = l2_loss(keypoints, gt_kps).mean()
        with torch.no_grad():
            reconstruction = self.decoder(keypoints).unsqueeze(1)
        out = {'keypoints': (127/2)*(keypoints + 1),
                'reconstruction':reconstruction,
                'loss':loss}
        return out
        
def eval_model(model, tgt_batch, kp_to_skl):
    model.eval()
    decoder = kp_to_skl
    images = tgt_batch['imgs']
    annots = tgt_batch['annots']
    with torch.no_grad():
        gt_skl = decoder(annots.to('cuda')).unsqueeze(1).detach()
        encoder_out = model(gt_skl, annots)
        loss = encoder_out['loss'].mean()
    model.train()
    return loss.item()

def debug_encoder(model_skeleton_to_keypoint,
                   model_keypoint_to_skeleton,
                   loader,
                   loader_tgt,
                   train_params,
                   checkpoint,
                   logger, device_ids, tgt_batch=None):
    log_params = train_params['log_params']
    optimizer_encoder = torch.optim.Adam(model_skeleton_to_keypoint.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    resume_epoch = 0
    resume_iteration = 0
    if checkpoint is not None:
        print('Loading Checkpoint: %s' % checkpoint)
        resume_epoch, resume_iteration = logger.checkpoint.load_checkpoint(checkpoint,
                                              skeleton_to_keypoints=model_skeleton_to_keypoint,
                                              optimizer_skeleton_to_keypoints=optimizer_encoder)
        logger.epoch = resume_epoch
        logger.iterations = resume_iteration

    scheduler_encoder = MultiStepLR(optimizer_encoder, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)

    encoder = SkeletonToKeypoints(model_skeleton_to_keypoint, 
                                   model_keypoint_to_skeleton)
    encoder = DataParallelWithCallback(encoder, device_ids=device_ids)


    
    with torch.no_grad():
        skeletons = model_keypoint_to_skeleton(tgt_batch['annots'].to('cuda')).unsqueeze(1).detach()
    tgt_batch = {'imgs': tgt_batch['imgs'],
                 'annots': tgt_batch['annots'],
                 'annots_unnormed': tgt_batch['annots'],
                 'skeletons': skeletons}
    k = 0
    for epoch in range(logger.epoch, train_params['num_epochs']):
        for i, batch  in enumerate(tqdm(loader)):
            annots = batch['annots'] 
            annots_gt = batch['annots_unnormed']

            with torch.no_grad():
                gt_skl = model_keypoint_to_skeleton(annots.to('cuda')).unsqueeze(1).detach() 
            encoder_out = encoder(gt_skl, annots)
            optimizer_encoder.zero_grad()
            loss = encoder_out['loss'].mean()
            loss.backward()

            optimizer_encoder.step()
            optimizer_encoder.zero_grad()
            ####### LOG VALIDATION
            if i % log_params['eval_frequency'] == 0:
                eval_loss = eval_model(encoder, next(iter(loader_tgt)), 
                                        model_keypoint_to_skeleton)
                eval_sz = int(len(loader)/log_params['eval_frequency'])
                it_number = epoch * eval_sz  + (logger.iterations/log_params['eval_frequency'])
                logger.add_scalar('Eval loss', eval_loss, it_number)

            ####### LOG
            logger.add_scalar('L2 loss', 
                               loss.item(), 
                               logger.iterations)
            if i in log_params['log_imgs']:
                with torch.no_grad():
                    encoder.eval()
                    target_out = encoder(tgt_batch['skeletons'], tgt_batch['annots'])
                    encoder.train()
                skl_out = target_out['reconstruction']
                kps_out = target_out['keypoints']

                concat_img = np.concatenate((draw_kp(tensor_to_image(tgt_batch['imgs'][k]),tgt_batch['annots_unnormed'][k]),
                                            tensor_to_image(tgt_batch['skeletons'][k]),
                                            tensor_to_image(skl_out[k]),
                                            draw_kp(tensor_to_image(tgt_batch['imgs'][k]), kps_out[k], color='red')), axis=2)
                concat_img_train = np.concatenate((
                                            tensor_to_image(gt_skl[k]),
                                            tensor_to_image(encoder_out['reconstruction'][k])),
                                            axis=2)
 
                logger.add_image('Eval_{%d}' % i, concat_img, epoch)
                logger.add_image('Train_{%d}' % i, concat_img_train, epoch)
                k += 1
                k = k % len(log_params['log_imgs']) 
            logger.step_it()

        scheduler_encoder.step()
        logger.step_epoch(models = {'skeleton_to_keypoints':model_skeleton_to_keypoint,
                                    'optimizer_skeleton_to_keypoints':optimizer_encoder})
