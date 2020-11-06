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
from modules.losses import masked_l2_loss ,masked_l2_heatmap_loss  
from modules.util import kp2gaussian2, gaussian2kp
from nips.utils import HeatMap
from nips.MTFAN import convertLayer
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw

class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.heatmap_res = self.detector.heatmap_res

    def forward(self, images, ground_truth, mask=None, kp_map=None):
        dict_out = self.detector(images)
        #print(f"dict_out[heatmaps] {dict_out['heatmaps'].shape}")
        
        #loss = masked_l2_loss(keypoints, ground_truth, mask)
        gt = ground_truth if kp_map is None else ground_truth[:, kp_map]
        #loss = masked_l2_loss(dict_out['heatmaps'], gt, mask)
        loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)
        kps = unnorm_kp(dict_out['value'])
        #print('heatmap size: ', dict_out['heatmaps'].shape)
        #print('kp out: ', dict_out['value'])
        #print('unnorm kps: ', unnorm_kp(dict_out['value']))
        return {"keypoints": kps,
                "heatmaps": dict_out['heatmaps'],
                "l2_loss": loss.mean(),
                }
        
def eval_model(model, tgt_batch, heatmap_res=122, hm_var=0.15):
    model.eval()
    images = tgt_batch['imgs']
    annots = tgt_batch['annots']
    gt_heatmaps = kp2gaussian2(annots, (heatmap_res, heatmap_res), hm_var).detach()
    mask = None if 'kp_mask' not in tgt_batch.keys() else tgt_batch['kp_mask']
    out = None
    with torch.no_grad():
        out = model(images, gt_heatmaps, mask)
        #out = model(images, annots, mask)
    model.train()
    return out

def train_kpdetector(model_kp_detector,
                       loader,
                       loader_tgt,
                       train_params,
                       checkpoint,
                       logger, device_ids, tgt_batch=None, kp_map=None):
    log_params = train_params['log_params']
    optimizer_kp_detector = torch.optim.Adam(model_kp_detector.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    resume_epoch = 0
    resume_iteration = 0
    if checkpoint is not None:
        print('Loading Checkpoint: %s' % checkpoint)
        # TODO: Implement Load/resumo kp_detector
        if train_params['test'] == False:
            resume_epoch, resume_iteration = logger.checkpoint.load_checkpoint(checkpoint,
                                                  model_kp_detector=model_kp_detector,
                                                  optimizer_kp_detector=optimizer_kp_detector)
        else:
            net_dict = model_kp_detector.state_dict()
            pretrained_dict = torch.load(checkpoint)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
            net_dict.update(pretrained_dict)
            model_kp_detector.load_state_dict(net_dict, strict=True)
            model_kp_detector.apply(convertLayer)
 
        logger.epoch = resume_epoch
        logger.iterations = resume_iteration
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)
    
    kp_detector = KPDetectorTrainer(model_kp_detector)
    kp_detector = DataParallelWithCallback(kp_detector, device_ids=device_ids)
    k = 0
    if train_params['test'] == True:
        results = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'])
        print(' MSE: ' + str(results['MSE']) + ' PCK: ' + str(results['PCK'])) 
        return

    heatmap_var = train_params['heatmap_var']

    for epoch in range(logger.epoch, train_params['num_epochs']):
        results = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'])
        results_train = evaluate(model_kp_detector, loader, dset=train_params['dataset']) 
        print('Epoch ' + str(epoch)+ ' MSE: ' + str(results['MSE']))
        logger.add_scalar('MSE test', results['MSE'], epoch)
        logger.add_scalar('PCK test', results['PCK'], epoch)
        logger.add_scalar('MSE train', results_train['MSE'], epoch)
        logger.add_scalar('PCK train', results_train['PCK'], epoch)
 
        for i, batch  in enumerate(tqdm(loader)):
            images = batch['imgs']
            if (images != images).sum() > 0:
                print('Images has NaN')
                break
            annots = batch['annots'] 
            gt_heatmaps = kp2gaussian2(annots, (model_kp_detector.heatmap_res, 
                                                model_kp_detector.heatmap_res), heatmap_var).detach() 
            if (annots != annots).sum() > 0 or (annots.abs() == float("Inf")).sum() > 0:
                print('Annotation with NaN')
                break
            mask = None if 'kp_mask' not in batch.keys() else batch['kp_mask']
            ######## REMOVE
            #print(f"b_mask {mask}")
            #print(f"mask {mask.shape}")
            ##################
            #kp_detector_out = kp_detector(images, annots, mask)
            kp_detector_out = kp_detector(images, gt_heatmaps, mask)

            loss = kp_detector_out['l2_loss'].mean()
            loss.backward()

            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()
            ####### LOG VALIDATION
            if i % log_params['eval_frequency'] == 0:
                tgt_batch = next(iter(loader_tgt))
                eval_out = eval_model(kp_detector, tgt_batch, model_kp_detector.heatmap_res, heatmap_var)
                eval_sz = int(len(loader)/log_params['eval_frequency'])
                it_number = epoch * eval_sz  + (logger.iterations/log_params['eval_frequency'])
                logger.add_scalar('Eval loss', eval_out['l2_loss'].mean(), it_number)
                concat_img = np.concatenate((draw_kp(tensor_to_image(tgt_batch['imgs'][k]),unnorm_kp(tgt_batch['annots'][k])),
                                            draw_kp(tensor_to_image(tgt_batch['imgs'][k]), eval_out['keypoints'][k], color='red')), axis=2)

                heatmap_img_0 = tensor_to_image(kp_detector_out['heatmaps'][k, 0].unsqueeze(0), True)
                heatmap_img_1 = tensor_to_image(kp_detector_out['heatmaps'][k, 5].unsqueeze(0), True)
                src_heatmap_0 = tensor_to_image(gt_heatmaps[k, 0].unsqueeze(0), True)
                src_heatmap_1 = tensor_to_image(gt_heatmaps[k, 5].unsqueeze(0), True)
                heatmaps_img = np.concatenate((heatmap_img_0, heatmap_img_1), axis = 2)
                src_heatmaps = np.concatenate((src_heatmap_0, src_heatmap_1), axis = 2)
 
                logger.add_image('Eval_', concat_img, logger.iterations)
                logger.add_image('heatmaps', heatmaps_img, logger.iterations)
                logger.add_image('src heatmaps', src_heatmaps, logger.iterations)
 
            ####### LOG
            logger.add_scalar('L2 loss', 
                               loss.item(), 
                               logger.iterations)
            if i in log_params['log_imgs']:
                concat_img_train = np.concatenate((draw_kp(tensor_to_image(images[k]), unnorm_kp(annots[k])),
                                                  draw_kp(tensor_to_image(images[k]), kp_detector_out['keypoints'][k], color='red')), axis=2)
 
                logger.add_image('Train_{%d}' % i, concat_img_train, logger.iterations)
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


