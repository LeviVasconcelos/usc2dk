from modules.networks import ImageToSkl, SklToKP, KPToSkl, ConditionalImageGenerator, Discriminator2D
from modules.losses import l2_loss, discriminator_gan_loss, generator_gan_loss
from datasets.humans36m import LoadHumansDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2
from sync_batchnorm import DataParallelWithCallback
import copy
from modules.vgg19 import Vgg19
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image, ImageDraw
from modules.losses import masked_l2_loss ,masked_l2_heatmap_loss, confidence_loss, masked_l2_heatmap_loss_kp
from modules.util import kp2gaussian2, gaussian2kp, gaussian2kp_v2


def batch_PCK(predicted_kp_, source_gt_, dset='mpii', threshold_=0.5, mask=None):
    '''
    Assumes predicted_kp and source_gt dimentions as: batch x kp x 2
    '''
    def unnorm_kp(kps):
         return (127./2.) * (kps + 1)
    pck_joints = {}
    pck_joints['mpii'] = [8, 9]
    pck_joints['lsp'] = [12, 13]
    pck_joints['penn'] = [0, 1]
    pck_joints['humans'] = [5, 6]
    threshold = {}
    threshold['penn'] = 0.5
    threshold['mpii'] = 0.5
    threshold['lsp'] = 0.5
    threshold['humans']= 0.5
    
    if 'mpii' in dset:
        dset = 'mpii'
    if 'h36m' in dset:
        dset = 'humans'
    if 'penn' in dset:
        dset = 'penn'

    predicted_kp = unnorm_kp(predicted_kp_)
    source_gt = unnorm_kp(source_gt_)
    metric = (source_gt[:, pck_joints[dset][0], :] - source_gt[:, pck_joints[dset][1], :]).norm(dim=1).unsqueeze(-1)
    distances = (predicted_kp - source_gt).norm(dim=2)
    distances_scaled = distances / metric
    if mask is None:
        mask = torch.ones(source_gt.shape[0], source_gt.shape[1], dtype=torch.float).to(predicted_kp.get_device())
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
    scores = distances_scaled <= threshold[dset]
    scores[mask == 0] = 0
    scores = scores.sum(dim=1).float()
    total_count = mask.sum(dim=1)
    return (scores/total_count).mean()

def batch_MSD(predicted_kp, gt_kp):
    ## kp size: batch_sz x kp_number x kp_dim
    kp_norms = torch.sqrt(torch.pow(predicted_kp - gt_kp, 2).sum(-1))
    #kp_norms size: batch_sz x kp_number
    mse = kp_norms.view(kp_norms.shape[0], -1).mean(-1)
    #mse [batch_sz]
    return mse.sum()

def batch_MSE(predicted_kp, gt_kp, mask=None):
    diff = torch.pow(predicted_kp - gt_kp, 2)
    if mask is not None:
       diff = diff * mask
       division = mask.sum(1).unsqueeze(-1)
       diff = (diff / division)
       diff = diff.view(diff.shape[0], -1).sum(-1)
       return diff.sum()
    kp_means = diff.view(diff.shape[0], -1).mean(-1)
    return kp_means.sum()

def evaluate(model, loader, dset='mpii', filter=None, device='cuda', reverse=False):
    model.eval()
    pck_scores = list()
    scores = list()
    count = 0.
    with torch.no_grad():
        for batch in tqdm(loader):
            out = model(batch['imgs'].to(device))
            try:
                if reverse:
                    mask = batch['kp_mask'][:,filter].to(device)
                else:
                    mask = batch['kp_mask'].to(device)
            except:
                mask = None
            if filter is not None:
                if reverse:
                    pck_score = batch_PCK(out['value'], batch['annots'][:,filter].to(device), dset=dset, mask=mask)
                    score = batch_MSE(out['value'], batch['annots'][:,filter].to(device), mask=mask)
                else:
                    pck_score = batch_PCK(out['value'][:,filter], batch['annots'].to(device), dset=dset, mask=mask)
                    score = batch_MSE(out['value'][:,filter], batch['annots'].to(device), mask=mask)
            else:
                pck_score = batch_PCK(out['value'], batch['annots'].to(device), dset=dset, mask=mask)
                score = batch_MSE(out['value'], batch['annots'].to(device), mask=mask) 
            scores.append(score)
            pck_scores.append(pck_score)
            count += batch['imgs'].shape[0]
    out = {
            'MSE':torch.Tensor(scores).sum()/count,
            'PCK':torch.Tensor(pck_scores).mean(),
            }
    model.train()
    return out

def batch_PCK_non_mean(predicted_kp_, source_gt_, dset='mpii', threshold_=0.5, mask=None):
    '''
    Assumes predicted_kp and source_gt dimentions as: batch x kp x 2
    '''
    def unnorm_kp(kps):
         return (127./2.) * (kps + 1)
    pck_joints = {}
    pck_joints['mpii'] = [8, 9]
    pck_joints['lsp'] = [12, 13]
    pck_joints['penn'] = [0, 1]
    pck_joints['humans'] = [5, 6]
    threshold = {}
    threshold['penn'] = 0.5
    threshold['mpii'] = 0.5
    threshold['lsp'] = 0.5
    threshold['humans']= 0.5
    
    if 'mpii' in dset:
        dset = 'mpii'
    if 'h36m' in dset:
        dset = 'humans'
    if 'penn' in dset:
        dset = 'penn'

    predicted_kp = unnorm_kp(predicted_kp_)
    source_gt = unnorm_kp(source_gt_)
    metric = (source_gt[:, pck_joints[dset][0], :] - source_gt[:, pck_joints[dset][1], :]).norm(dim=1).unsqueeze(-1)
    distances = (predicted_kp - source_gt).norm(dim=2)
    distances_scaled = distances / metric
    if mask is None:
        mask = torch.ones(source_gt.shape[0], source_gt.shape[1], dtype=torch.float).to(predicted_kp.get_device())
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
    scores = distances_scaled <= threshold[dset]
    scores[mask == 0] = 0
    scores = scores.sum(dim=1).float()
    total_count = mask.sum(dim=1)
    return scores/total_count


def evaluate_with_loss(model, loader, dset='mpii', filter=None, device='cuda', reverse=False, heatmap_var=0.15):
    model.eval()
    confidence_scores = 0
    losses_real  = torch.Tensor([]).to(device)
    losses_conf = torch.Tensor([]).to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            img = batch['imgs'].to(device)
            out = model(img)
            try:
                if reverse:
                    mask = batch['kp_mask'][:,filter].to(device)
                else:
                    mask = batch['kp_mask'].to(device)
            except:
                mask = None

            if filter is not None:
                if reverse:
                    pck_score = batch_PCK_non_mean(out['value'], batch['annots'][:,filter].to(device), dset=dset, mask=mask)
                else:
                    gt_heatmaps = kp2gaussian2(batch['annots'].to(device), (model.heatmap_res,model.heatmap_res), heatmap_var).detach() 
                    loss = masked_l2_heatmap_loss(out['heatmaps'][:,filter],gt_heatmaps, mask)
                    losses_real = torch.cat((losses_real, loss),0)
                    losses_conf = torch.cat((losses_conf, out["confidence"].mean(1)),0)
            else:
                gt_heatmaps = kp2gaussian2(batch['annots'].to(device), (model.heatmap_res, 
                                     model.heatmap_res), heatmap_var).detach() 
                loss = masked_l2_heatmap_loss(out['heatmaps'],gt_heatmaps, mask)
                losses_real = torch.cat((losses_real, loss),0)
                losses_conf = torch.cat((losses_conf, out["confidence"].mean(1)),0)

    losses_conf =  losses_conf.cpu()
    losses_real = losses_real.cpu()
    indexes = np.where(losses_conf < np.percentile(losses_conf,5))[0]
    real_index = np.where(losses_real < np.percentile(losses_real,5))[0]
    #print(confidence_scores/total_samples)            
    model.train()
    return losses_real[indexes].mean(), losses_real[real_index].mean()

def evaluate_confidence(model, loader, dset='mpii', filter=None, device='cuda', reverse=False):
    model.eval()
    confidence_scores = 0
    total_samples = 0 
    with torch.no_grad():
        for batch in tqdm(loader):
            total_samples += batch['imgs'].shape[0]
            out = model(batch['imgs'].to(device))
            try:
                if reverse:
                    mask = batch['kp_mask'][:,filter].to(device)
                else:
                    mask = batch['kp_mask'].to(device)
            except:
                mask = None

            if filter is not None:
                if reverse:
                    pck_score = batch_PCK_non_mean(out['value'], batch['annots'][:,filter].to(device), dset=dset, mask=mask)
                else:
                    pck_score = batch_PCK_non_mean(out['value'][:,filter], batch['annots'].to(device), dset=dset, mask=mask)
            else:
                pck_score = batch_PCK_non_mean(out['value'], batch['annots'].to(device), dset=dset, mask=mask)

            indexes = np.where(np.abs(out["confidence"].squeeze(-1).cpu()-pck_score.cpu()) < 0.01)[0]
            confidence_scores += len(indexes)
    #print(confidence_scores/total_samples)            
    model.train()
    return round(confidence_scores/total_samples, 4)







