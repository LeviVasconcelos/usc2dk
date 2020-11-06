import os
import sys
import yaml
from time import gmtime, strftime
from argparse import ArgumentParser
from shutil import copy
import torch
import json
from PIL import Image, ImageDraw
import matplotlib

import pickle as pkl
import numpy as np
from tqdm import tqdm


from evaluation import batch_PCK, batch_MSE
from modules.networks import KPToSkl, Discriminator2D
from modules.networks import MultiScaleDiscriminator 
from modules.networks import KPDetectorVerbose, KPDetector

from datasets.humans36m import LoadHumansDataset, batch_fn
from datasets.penn_action import LoadPennAction, batch_penn
from datasets.mpii_loader import LoadMpii
from datasets.couple_loader import LoadCoupledDatasets
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.lsp import LoadLsp
from datasets.annot_converter import HUMANS_TO_LSP, HUMANS_TO_MPII, HUMANS_TO_PENN, MPII_TO_HUMANS
from modules.util import kp2gaussian2
from kp_disc_geo import train_generator_geo

from tensor_logger import Logger

def evaluate(model, loader, dset='mpii', filter=None, device='cuda'):
    model.eval()
    pck_scores = list()
    scores = list()
    concat_imgs = []
    count = 0.
    k = 1
    with torch.no_grad():
        for batch in tqdm(loader):
            out = model(batch['imgs'].to(device))
            try:
                mask = batch['kp_mask'].to(device)
            except:
                mask = None
            if filter is not None:
                pck_score = batch_PCK(out['value'][:,filter], batch['annots'].to(device), dset=dset, mask=mask)
                score = batch_MSE(out['value'][:,filter], batch['annots'].to(device), mask=mask)
            else:
                pck_score = batch_PCK(out['value'], batch['annots'].to(device), dset=dset, mask=mask)
                score = batch_MSE(out['value'], batch['annots'].to(device), mask=mask) 
            scores.append(score)
            pck_scores.append(pck_score)
            count += batch['imgs'].shape[0]

            #eval_out = eval_model(model, batch)
            #print(f"annots {batch['annots'][k].shape}")
            #print(f"kp {out['value'][k].shape}")
            #print(f"img {batch['imgs'][k].shape}")
            
            concat_img = np.concatenate((draw_kp(tensor_to_image(batch['imgs'][k]),unnorm_kp(batch['annots'][k])),
                                            draw_kp(tensor_to_image(batch['imgs'][k]), unnorm_kp(out['value'][k]), color='red')), axis=2)
            concat_imgs.append(concat_img)

    mse = torch.Tensor(scores).sum()/count
    pck = torch.Tensor(pck_scores).mean()
    out = {
            'MSE':mse.item(),
            'PCK':pck.item(),
            }
    model.train()
    return out, concat_imgs

def eval_model(model, tgt_batch, heatmap_res=122):
    model.eval()
    images = tgt_batch['imgs']
    annots = tgt_batch['annots']
    gt_heatmaps = kp2gaussian2(annots, (heatmap_res, heatmap_res), 0.15)
    mask = None if 'kp_mask' not in tgt_batch.keys() else tgt_batch['kp_mask']
    out = None
    with torch.no_grad():
        out = model(images, gt_heatmaps, mask)
        #out = model(images, annots, mask)
    return out

def draw_kp(img_, kps, color='blue'):
    #print(img_.shape)
    #print(kps.shape)
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

def tensor_to_image(x):
    out = x.clone().detach().cpu()
    out = out.numpy()
    #out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    out = (out * 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--src_model", required=True, help="Model to adapt")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--tgt", default='humans')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
    
    ##### Model instantiation
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 
    kp_state_dict = torch.load(opt.src_model)

    if opt.src_model != 'scratch':
        try:
            print(f"loading {opt.src_model}")
            kp_state_dict = torch.load(opt.src_model)

            print('Source model loaded: %s' % opt.src_model)
        except:
            print('Failed to read model %s' % opt.src_model)
            exit(1)
        try:
            model_kp_detector.load_state_dict(kp_state_dict['model_kp_detector'])
        except:
            print('failed to load model weights')
            exit(1)

    disc_params = config['model_params']['discriminator_params']
    obj_edges = opt.tgt

    ##### Dataset loading
    config['train_params']['dataset'] = opt.tgt
    if opt.tgt == 'penn':
        loader_test = LoadPennAction(**config['datasets']['penn_test'])
        kp_map = HUMANS_TO_PENN
    elif opt.tgt == 'mpii':
        loader_test = LoadMpii(**config['datasets']['mpii_eval'])
        kp_map = HUMANS_TO_MPII
    elif opt.tgt == 'lsp':
        loader_test = LoadLsp(**config['datasets']['lsp_test'])
        kp_map = HUMANS_TO_LSP
    elif opt.tgt == 'humans':
        loader_test = LoadHumansDataset(**config['datasets']['h36m_resized_simplified_test'])
        kp_map =  MPII_TO_HUMANS # #
        # kp_map = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 19, 25, 26, 27]

    results, imgs = evaluate(model_kp_detector, loader_test, filter=kp_map,dset=opt.tgt)
    
    pkl.dump(imgs, open(os.path.join(os.path.split(opt.src_model)[0],"img.pkl"), 'wb'))
    print(f" res : {results}")

    json.dump(results,open(os.path.join(os.path.split(opt.src_model)[0],"evaluation.json"), "w"),indent=2)
