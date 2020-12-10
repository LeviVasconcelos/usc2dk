import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from evaluation import evaluate, evaluate_confidence, evaluate_with_loss
from sync_batchnorm import DataParallelWithCallback 
from modules.losses import masked_l2_loss ,masked_l2_heatmap_loss, confidence_loss, masked_l2_heatmap_loss_kp
from modules.util import kp2gaussian2, gaussian2kp, gaussian2kp_v2
from nips.utils import HeatMap
from nips.MTFAN import convertLayer
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw
import torch.nn.functional as F



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
    return scores/total_count

class KPDetectorTrainer(nn.Module):
    def __init__(self, kp_detector):
        super(KPDetectorTrainer, self).__init__()
        self.detector = kp_detector
        self.heatmap_res = self.detector.heatmap_res
        self.confidence_loss =  confidence_loss

    def forward(self, images, ground_truth, mask=None, kp_map=None, kp_loss=False, pretrained=False,dset=None, evaluate=None, device="cuda"):
        #print(evaluate)
        if mask is not None:
                mask = mask.to(device)
                
        if evaluate:
            with torch.no_grad():
                dict_out = self.detector(images)
                dict_out["value"] = dict_out["value"] if kp_map is None else dict_out["value"][:, kp_map]
                dict_out["heatmaps"] = dict_out["heatmaps"] if kp_map is None else dict_out["heatmaps"][:, kp_map]
                #dict_out["confidence"] = dict_out["confidence"] if kp_map is None else dict_out["confidence"][:, kp_map]

                kp_map = kp_map

                gt = ground_truth #if kp_map is None else ground_truth[:, kp_map]
                # print(dict_out["value"].shape)
                # print(gt.shape)
                if not pretrained:
                    if kp_loss:
                        loss = masked_l2_loss(dict_out['value'], gt.detach(), mask)
                    else:
                        loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)
                    loss = loss.mean()
                else:
                    loss = 0
                #print(mask.shape)
                #print(mask)
                #mask[mask != mask] = 0
              
                #loss = masked_l2_heatmap_loss_kp(dict_out['heatmaps'], gt.detach(), mask)
                loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)

                #print(loss.shape)
                #print(dict_out['confidence'].shape)
                #print(loss.shape)
                #print(dict_out['confidence'].shape)

                # if mask is not None:
                #     dict_out['confidence'] = dict_out['confidence'] * mask.squeeze(-1) 
                
                confidence_loss = F.binary_cross_entropy_with_logits(dict_out['confidence'].mean(1), loss)

                kps = unnorm_kp(dict_out['value'])
                # gt_kp = gaussian2kp_v2(ground_truth)["mean"]
                # #gt_kp = gt_kp if kp_map is None else gt_kp[:, kp_map]
                # pck_l  = batch_PCK(dict_out["value"],gt_kp,dset=dset, mask=mask)
                # confidence_loss = self.confidence_loss(dict_out['confidence'].squeeze(-1), pck_l, gamma=0.5, alpha=0.5).mean()

        else:
            dict_out = self.detector(images)
            gt = ground_truth
            #gt = ground_truth if kp_map is None else ground_truth[:, kp_map]
            dict_out["value"] = dict_out["value"] if kp_map is None else dict_out["value"][:, kp_map]
            dict_out["heatmaps"] = dict_out["heatmaps"] if kp_map is None else dict_out["heatmaps"][:, kp_map]
            dict_out["confidence"] = dict_out["confidence"] if kp_map is None else dict_out["confidence"][:, kp_map]

            if not pretrained:
                if kp_loss:
                    loss = masked_l2_loss(dict_out['value'], gt.detach(), mask)
                else:
                    loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)
                loss = loss.mean()
            else:
                loss = 0

            kps = unnorm_kp(dict_out['value'])
            #pck_l  = batch_PCK(dict_out["value"], gaussian2kp_v2(gt)["mean"], mask=mask)

            with torch.no_grad():
                #loss = masked_l2_heatmap_loss_kp(dict_out['heatmaps'], gt.detach(), mask)
                loss = masked_l2_heatmap_loss(dict_out['heatmaps'], gt.detach(), mask)

            # if mask is not None:
            #     dict_out['confidence'] = dict_out['confidence']* mask.squeeze(-1) 


            confidence_loss = F.binary_cross_entropy_with_logits(dict_out['confidence'].mean(1), loss)

            #confidence_loss = self.confidence_loss(dict_out['confidence'].squeeze(-1), pck_l, gamma=0.5, alpha=0.5).mean()

        return {"keypoints": kps,
                "heatmaps": dict_out['heatmaps'],
                "l2_loss": loss,
                "confidence_l" : confidence_loss
                }

        
def eval_model(model, tgt_batch, heatmap_res=122, hm_var=0.15, kp_loss=True):
    model.eval()
    images = tgt_batch['imgs']
    annots = tgt_batch['annots']
    gt_heatmaps = kp2gaussian2(annots, (heatmap_res, heatmap_res), hm_var).detach()
    mask = None if 'kp_mask' not in tgt_batch.keys() else tgt_batch['kp_mask']
    out = None
    with torch.no_grad():
        if kp_loss:
            out = model(images, annots, mask, kp_loss)
        else:
            out = model(images, gt_heatmaps, mask, kp_loss)

        #out = model(images, annots, mask)
    model.train()
    return out

def train_kpdetector(model_kp_detector,
                       loader,
                       loader_tgt,
                       train_params,
                       checkpoint,
                       logger, device_ids, tgt_batch=None, kp_map=None, pretrained=False, loader_third=None):
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

    iterator_tgt = iter(loader_tgt)
    iterator_third = iter(loader_third)


    heatmap_var = train_params['heatmap_var']
    kp_loss = train_params["kp_loss"]
    print(f"kp_loss {kp_loss}")
    device_ids = device_ids[0]
    for epoch in range(logger.epoch, train_params['num_epochs']):
        if loader_third is not None:
            results_train, real_treshold = evaluate_with_loss(model_kp_detector, loader_third, dset="penn", device=device_ids, filter=kp_map, heatmap_var=heatmap_var)
            print('Epoch ' + str(epoch)+ 'target domain --> mean estimated: ' + str(results_train) + ' real mean: ' + str(real_treshold) )
            logger.add_scalar('target train', results_train, epoch)

        # else:
        #     results = evaluate_with_loss(model_kp_detector, loader_tgt, dset="mpii", device=device_ids, filter=kp_map, heatmap_var=heatmap_var)

        results_train, real_treshold  = evaluate_with_loss(model_kp_detector, loader, dset="mpii", device=device_ids, heatmap_var=heatmap_var) 
        print('Epoch ' + str(epoch)+ 'source domain train --> mean estimated: ' + str(results_train) + ' real mean: ' + str(real_treshold) )
        logger.add_scalar('est train', results_train, epoch)

        results_train, real_treshold  = evaluate_with_loss(model_kp_detector, loader_tgt, dset="mpii", device=device_ids, heatmap_var=heatmap_var) 
        print('Epoch ' + str(epoch)+ 'source domain eval --> mean estimated: ' + str(results_train) + ' real mean: ' + str(real_treshold) )
        logger.add_scalar('val train', results_train, epoch)

        # #results_tgt = evaluate(model_kp_detector, loader_tgt, dset=train_params['dataset'], filter=kp_map, device=device_ids)
        # #print('Epoch ' + str(epoch)+ ' PCK: ' + str(results_tgt["PCK"]))
        # print('Epoch ' + str(epoch)+ ' source percentage: ' + str(results_train))
        #logger.add_scalar('perc test', results_train, epoch)
 
        for i, batch  in enumerate(tqdm(loader)):
            images = batch['imgs']
            try:
                tgt_batch = next(iterator_tgt)
            except:
                iterator_tgt = iter(loader_tgt)
                tgt_batch = next(iterator_tgt)
            try:
                third_batch = next(iterator_third)
            except:
                iterator_third = iter(loader_third)
                third_batch = next(iterator_third)

            if (images != images).sum() > 0:
                print('Images has NaN')
                break
            
            annots = batch['annots'] 
            #print(f"shape annotsh  : {annots.shape}")
            
            gt_heatmaps = kp2gaussian2(annots, (model_kp_detector.heatmap_res, 
                                                model_kp_detector.heatmap_res), heatmap_var).detach() 
            if (annots != annots).sum() > 0 or (annots.abs() == float("Inf")).sum() > 0:
                print('Annotation with NaN')
                break
            mask = None if 'kp_mask' not in batch.keys() else batch['kp_mask']

            if kp_loss:
                kp_detector_out = kp_detector(images, annots, mask, kp_loss=kp_loss, pretrained=pretrained)
            else:
                kp_detector_out = kp_detector(images, gt_heatmaps, mask, kp_loss=kp_loss, pretrained=pretrained, evaluate=False, dset="mpii", device=device_ids)
            loss_r = 0
            if not  pretrained:
                loss_r = kp_detector_out['l2_loss'].mean()
            
            loss_c = train_params["loss_weights"]["confidence"] * kp_detector_out["confidence_l"]
            # loss = loss_c + loss_r
            loss_c.backward()

            if loader_third is not None:
                mask = None if 'kp_mask' not in batch.keys() else tgt_batch['kp_mask']
                images = tgt_batch["imgs"]
                annots = tgt_batch['annots'] 
                gt_heatmaps = kp2gaussian2(annots, (model_kp_detector.heatmap_res, 
                                     model_kp_detector.heatmap_res), heatmap_var).detach() 
                kp_detector_out_third =  kp_detector(images, gt_heatmaps, mask, kp_loss=kp_loss, pretrained=pretrained,evaluate=False, dset="mpii",  device=device_ids)
                loss_c_eval = train_params["loss_weights"]["confidence"] * kp_detector_out_third["confidence_l"]
                
                #loss_c_eval.backward()

                images = third_batch["imgs"]
                annots = third_batch['annots'] 
                mask = None if 'kp_mask' not in batch.keys() else third_batch['kp_mask']
                gt_heatmaps = kp2gaussian2(annots, (model_kp_detector.heatmap_res, 
                                     model_kp_detector.heatmap_res), heatmap_var).detach() 
                kp_detector_out_third =  kp_detector(images, gt_heatmaps, mask, kp_loss=kp_loss, pretrained=pretrained,evaluate=True, kp_map=kp_map, dset="penn", device=device_ids)

                loss_target = train_params["loss_weights"]["confidence"] * kp_detector_out_third["confidence_l"]
                
                #loss_target.backward()
                # gt_heatmaps = kp2gaussian2(annots, (model_kp_detector.heatmap_res, 
                #                     model_kp_detector.heatmap_res), heatmap_var).detach() 
                # if kp_loss:
                #     kp_detector_out = kp_detector(images, annots, mask, kp_loss=kp_loss, pretrained=pretrained)
                # else:
                #     kp_detector_out = kp_detector(images, gt_heatmaps, mask, kp_loss=kp_loss, pretrained=pretrained)
                # loss_c_eval = train_params["loss_weights"]["confidence"] * kp_detector_out["confidence_l"]
                # loss_eval = loss_c_eval 
                # loss_eval.backward()
            # loss_c = train_params["loss_weights"]["confidence"] * kp_detector_out["confidence_l"]
            # loss = loss_c + loss_r
            # loss.backward()

            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()

            ####### LOG
            logger.add_scalar('L2 loss', 
                               loss_r, 
                               logger.iterations)

            # logger.add_scalar('sum loss', 
            #                    loss.item(), 
            #                    logger.iterations)

            logger.add_scalar('Confidence loss', 
                               loss_c.item(), 
                               logger.iterations)
            if loader_third is not None:

                logger.add_scalar('Confidence Eval loss', 
                                loss_c_eval.item(), 
                                logger.iterations)

                logger.add_scalar('Confidence Target loss', 
                                loss_target, 
                                logger.iterations)


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


