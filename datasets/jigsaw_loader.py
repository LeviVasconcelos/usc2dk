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


class Humans36mJigSawDataset(Humans36mDataset):
    def __init__(self, nViews, split='train', rgb=True,
                 nPerSubject=2000, subjects=[0],
                 meta_val=1, img_size=224, normalized=True, gt_only=False,
                 permutations_file=None):
        super(Humans36mJigSawDataset, self).__init__(nViews, split, rgb, nPerSubject, subjects,
                 meta_val, img_size, normalized, gt_only)

        self.permutations = np.load(permutations_file)
        # from range [1,9] to [0,8]
        if self.permutations.min() == 1:
            self.permutations = self.permutationst - 1

        self._augment_tile = lambda x: x
        self.grid_size=4

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile


    def __getitem__(self, index):
        idx = index % self.len

        imgs = np.zeros((self.nViews, self.img_size, self.img_size, 3), dtype=np.float32)
        ref_imgs = np.zeros((self.nViews, self.img_size, self.img_size, 3), dtype=np.float32)
        annots = np.zeros((self.nViews, self.K, 2), dtype=np.float32)
        ref_annots = []
        pose_2d = []
        dataset_ref_idx = self._load_ref_idx(idx)
        ratio = float(self.original_size / self.img_size)
        ref_img_type = 'Views' if self.rgb else 'TOF'
        for k in range(self.nViews):
            imgs[k] = self._transform_image(self._load_image(idx, k), self._get_ref(idx), k)

            n_grids = self.grid_size ** 2
            tiles = [None] * n_grids
            for n in range(n_grids):
                tiles[n] = self.get_tile(imgs[k], n)

            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted

            if order == 0:
                data = tiles
            else:
                data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

            ref_imgs[k] = torch.stack(data, 0)
            ref_annots[k]  = int(order)

            if self.rgb:
                annots[k] = self._transform_2dkp_annots(imgs[k].shape, self._get_ref(idx), k)
            else:
                annots[k] = self._transform_2dkp_annots(imgs[k].shape, self._get_ref(idx), 1)

        inp = None
        if not self.gt_only:
            imgs = imgs.transpose(0, 3, 1, 2) / 255.
            ref_imgs = ref_imgs.transpose(0, 3, 1, 2) / 255.
            inp = torch.from_numpy(imgs)
            ref_imgs = torch.from_numpy(ref_imgs)
        return inp, annots, ref_imgs, ref_annots

    def __len__(self):
        return self.len
