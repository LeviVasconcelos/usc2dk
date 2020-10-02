#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:30:29 2019
@author: levi
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as TF
import h5py
from PIL import Image

from tqdm import tqdm
import pickle
from .h36m_metadata import H36M_Metadata
import numpy as np
from datasets.humans36m import *
import random

# Humans_dir = '/hd/levi/workspace/h36m-fetch/processed/'
Humans_dir = '/hd/levi/workspace/h36m-fetch/video_processed_fullres/'
metaDim = 20


def batch_fn(data):
    img_size=128
    def shift_to_interval(x):
        '''
        Make return \in [-1, 1], assuming x \in [0,1]
        '''
        return (2*x - 1)

    annot = list(data)
    annots = torch.from_numpy(np.stack(annot, 0)) / (img_size -1)

    out = {
            "annots":shift_to_interval(annots),
            "annots_unnormed": annots * (img_size-1),
            "kp_mask": None,
            }
    return out 

def LoadSkeletonDataset(subjects, nImages, 
                         root_dir, kp_map,  
                         batch_size=1, workers=16, process_batch=True):
    #KP LIST SHOULD BE ALL 32 KPS
    h36m = Humans36mSkeletons(nViews=4, root_dir=root_dir, 
                              subjects=subjects,  kp_map=kp_map,
                              nPerSubject=nImages)
    return torch.utils.data.DataLoader(h36m, batch_size=batch_size, 
                                 shuffle=True, num_workers=workers, 
                                 pin_memory=False, collate_fn=batch_fn)
 

class Humans36mSkeletons(Humans36mDataset):
    def __init__(self, nViews, root_dir, 
                 nPerSubject=2000, subjects=[0], 
                  kp_map=None):
        self.kp_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        super(Humans36mSkeletons, self).__init__(nViews, root_dir, '', 
                                                  rgb=True, nPerSubject=nPerSubject, 
                                                  subjects=subjects, kp_list=self.kp_list,
                                                  meta_val=-1, img_size=128, 
                                                  normalized=False, gt_only=True)
        self.kp_map = kp_map

    def __getitem__(self, idx):
        view = random.randint(0, self.nViews -1)
        if self.kp_map is not None:
            kps = self._get_ref(idx)['Annot']['2d'][view][self.kp_map] 
        else:
            kps = self._get_ref(idx)['Annot']['2d'][view]
        return kps


 
