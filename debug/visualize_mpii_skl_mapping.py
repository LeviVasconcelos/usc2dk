import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
import yaml
import cv2
import numpy as np

from datasets.mpii_loader import LoadMpii 
from datasets.humans36m import LoadHumansDataset, batch_fn

from datasets.annot_converter import MPII_TO_HUMANS, skeletons
from PIL import Image, ImageDraw

from modules.networks import KPToSkl 
from utils import tensor_to_image, draw_kp


kConfig = 'config/u2dkp.yaml'
with open(kConfig) as f:
   config = yaml.load(f)

kp2skl_params = config['model_params']['KTS_params']
mpii_params = kp2skl_params
mpii_params['n_kps'] = 16
mpii_params['edges'] = skeletons['mpii']

h36m_params = kp2skl_params
h36m_params['n_kps'] = 15
h36m_params['edges'] = skeletons['humans']
kp2skl_mpii = KPToSkl(**mpii_params)#.to(opt.device_ids[0])
kp2skl_h36m = KPToSkl(**h36m_params)#.to(opt.device_ids[0])

dset_loader = LoadMpii(**config['datasets']['mpii_eval'])
h36m_loader = LoadHumansDataset(**config['datasets']['h36m_resized_crop_protocol2_train'])

batch = next(iter(dset_loader))
h36m_batch = next(iter(h36m_loader))
imgs = batch['imgs']
gt = batch['annots']
gt_mapped = batch['annots'][:, MPII_TO_HUMANS]

mpii_skl = kp2skl_mpii(gt)
mpii_map_skl = kp2skl_h36m(gt_mapped)
idx = 0
skl_img_mpii = tensor_to_image(mpii_skl[idx].unsqueeze(0))
skl_img_humans = tensor_to_image(mpii_map_skl[idx].unsqueeze(0))

img = tensor_to_image(imgs[idx])
img_mpii = draw_kp(img, gt[idx])
img_mpii_map = draw_kp(img, gt_mapped[idx])

skl_img = np.concatenate((skl_img_mpii, skl_img_humans),axis=2)
skl_img = np.transpose(skl_img, (1,2,0))
img_cat = np.concatenate((img_mpii, img_mpii_map), axis=2)
img_cat = np.transpose(img_cat, (1,2,0))

Image.fromarray(skl_img).save('skel_demo.png', 'PNG')
Image.fromarray(img_cat).save('img_demo.png', 'PNG')


h36m_img = h36m_batch['imgs']
h36m_gt = h36m_batch['annots']
h_skl = kp2skl_h36m(h36m_gt)
h_skl_img = tensor_to_image(h_skl[idx].unsqueeze(0))
h_skl_img = np.transpose(h_skl_img, (1,2,0))

h_img = tensor_to_image(h36m_img[idx])
h_img = draw_kp(h_img, h36m_gt[idx])
h_img = np.transpose(h_img, (1,2,0))
Image.fromarray(h_img).save('h_img.png', 'PNG')
Image.fromarray(h_skl_img).save('h_skl.png', 'PNG')

