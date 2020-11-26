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
from torchvision.transforms import ColorJitter, ToTensor, ToPILImage
import h5py
from PIL import Image

from tqdm import tqdm
import pickle
from .h36m_metadata import H36M_Metadata

#Humans_dir = '/hd/levi/workspace/h36m-fetch/processed/'
Humans_dir = '/hd/levi/workspace/h36m-fetch/video_processed_resized_crop'
#Humans_dir = '/hd/levi/workspace/h36m-fetch/video_processed_simplified'
#Humans_dir = '/hd/levi/workspace/h36m-fetch/resized_crop_full_dataset/'
#Humans_dir = '/hd/levi/workspace/h36m-fetch/resized_crop_simplified_full_dataset/'
metaDim = 20
def batch_fn(data):
    def flatten_views(img):
        return img.view(-1, *img.shape[2:])
    def shift_to_interval(x):
        '''
        Make return \in [-1, 1], assuming x \in [0,1]
        '''
        return (2*x - 1)

    ziped_data = list(zip(*data))
    (img, annot, ref_img, ref_annot) = list(zip(*data))
    imgs = flatten_views(torch.stack(img, 0))
    ref_imgs = flatten_views(torch.stack(ref_img, 0))
    annots = flatten_views(torch.from_numpy(np.stack(annot, 0))) / (imgs.shape[2]-1)
    ref_annots = flatten_views(torch.from_numpy(np.stack(ref_annot, 0))) / (imgs.shape[2]-1)

    out = {
            "imgs":imgs,
            "ref_imgs":ref_imgs,
            "ref_annots":shift_to_interval(ref_annots),
            "annots":shift_to_interval(annots),
            "annots_unnormed": annots * (imgs.shape[2]-1),
            "kp_mask": None,
            }
    return out 

def LoadHumansDataset(subjects, rgb, nImages, nViews, root_dir, kp_list,  batch_size=1, img_size=128, workers=16, process_batch=True, color_jitter=None):
    h36m =  Humans36mDataset(nViews, root_dir, '', rgb, 
                              nImages, subjects,  kp_list, -1, normalized=False, img_size=img_size, color_jitter=color_jitter)
    if process_batch:
        return torch.utils.data.DataLoader(h36m, batch_size=batch_size, 
                                     shuffle=True, num_workers=workers, 
                                     pin_memory=False, collate_fn=batch_fn, drop_last=True)
    else:
        return torch.utils.data.DataLoader(h36m, batch_size=batch_size, 
                                              shuffle=True, num_workers=workers, 
                                              pin_memory=False, drop_last=True)

def _draw_annot_from_file(img, bbox, pose):
      img2 = cv2.imread(img)
      cv2.rectangle(img2, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 3)
      for i in pose:
            cv2.circle(img2, tuple(i), 1, (255,0,0), -1)
      return img2

def _draw_annot(img, pose):
      print(img.shape)
      img2 = img.copy()
      for i in pose:
            cv2.circle(img2, tuple(i), 1, (255,0,0), -1)
      return img2

def Humans36mRGBSourceDataset(split, nViews, nImages=2000, normalized=True, meta=1):
      subjects = [0, 1, 2] if split == 'train' else [5,6]
      return Humans36mDataset(nViews, split, True, nImages, subjects, meta_val=meta, normalized=normalized)

def Humans36mRGBTargetDataset(split, nViews, nImages=2000, normalized=True, meta=5):
      subjects = [3, 4] if split == 'train' else [5,6]
      if split != 'train':
          meta *= -1
      return Humans36mDataset(nViews, split, True, nImages, subjects, meta_val=meta, normalized=normalized)


def Humans36mDepthSourceDataset(split, nViews, nImages=2000, normalized=True, meta=1):
      subjects = [0, 1, 2] if split == 'train' else [5,6]
      return Humans36mDataset(1, split, False, nImages, subjects, meta_val=meta, normalized=normalized)

def Humans36mDepthTargetDataset(split, nViews, nImages=2000, normalized=True, meta=5):
      subjects = [3, 4] if split == 'train' else [5,6]
      if split != 'train':
          meta *= -1
      return Humans36mDataset(1, split, False, nImages, subjects, meta_val=meta, normalized=normalized)


class Humans36mDataset(data.Dataset):
      def __init__(self, nViews, root_dir, split='train', rgb=True, 
                    nPerSubject=2000, subjects = [0],
                    kp_list = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 19, 25, 26, 27],
                    meta_val=1, img_size=224, normalized=False, gt_only=False, 
                    ref_interval=[3,30], color_jitter=None):
            self.ref_interval = ref_interval
            self.original_size = 224
            self.normalized = normalized
            self.gt_only = gt_only
            self.root_dir = root_dir
            self.rgb = rgb
            self.nViews = nViews if self.rgb else 1
            self.split = split
            self.ColorJitter = None
            if color_jitter is not None:
                self.ColorJitter = ColorJitter(brightness = 0,
                                               contrast=0,
                                               saturation=0,
                                               hue=0.5)
            self.kp_to_use = np.asarray(kp_list)
            self.K = len(self.kp_to_use)
            self.meta_val = meta_val
            self.metadata = H36M_Metadata(os.path.join(self.root_dir, 'metadata.xml'))
            self.imagesPerSubject = nPerSubject
            self.kBlacklist = { 
                        ('S11', '2', '2'),  # Video file is corrupted
                        ('S7', '15', '2'), # TOF video does not exists.
                        ('S5', '4', '2'), # TOF video does not exists.
                        }
            kSubjects = np.asarray(['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'])
            self.subjects_to_include = kSubjects[subjects]
            self.kFolders = {
                        'rgb_cameras' : 'imageSequence',
                        'tof_data' : 'ToFSequence'
                        }
            self.kCameras = [str(x) for x in self.metadata.camera_ids]
            self.kMaxViews = len(self.kCameras)
            self.kAnnotationsFilename = 'annot.h5'
            print('building dataset')
            self._build_indexes()
            self._build_access_index()
            self._build_meta()
            self.img_size = img_size 
            self.resize_transform = torchvision.transforms.Resize(self.img_size)
            self._normalization = self._normalize_pose if self.normalized else (lambda a : a)
            print('**Dataset Loaded: split [%s], len[%d], views[%d], rgb[%s]' % 
                  (self.split, self.len, self.nViews, 'True' if self.rgb else 'False'))
      
      def _generate_instance_filename(self):
            str_kps = ['%d' % i for i in self.kp_to_use.tolist()]
            filename = 'Humans' + ('RGB' if self.rgb else 'Depth') + '_' + \
                       '-'.join(str_kps) + '_subjescts_' + \
                       '-'.join(self.subjects_to_include.tolist()) + '_split_' + self.split + \
                       '_nPerSubject_' + str(self.imagesPerSubject) + '.pkl'
            return filename

      def _load_dataindex(self):
            filename = os.path.join(self.root_dir, 'cache', self._generate_instance_filename())
            if not os.path.isfile(filename):
                return False
            print('loading dataset, from cache...')
            try:
                with open(filename, 'rb') as dataset_pickle_file: 
                    self.dataset_indexes = pickle.load(dataset_pickle_file)
                    self.access_order = pickle.load(dataset_pickle_file)
                    self.meta = pickle.load(dataset_pickle_file)
                    self.poses_mean, self.poses_std = pickle.load(dataset_pickle_file)
                    self.len = len(self.access_order)
                    self.nImages = self.len * self.nViews
                    return True
            except EOFError as e:
                return False
            return False

      def _save_dataindex(self):
            filename = os.path.join(self.root_dir, 'cache', self._generate_instance_filename())
            with open(filename, 'wb') as dataset_pickle_file: 
                  pickle.dump(self.dataset_indexes, dataset_pickle_file) # dataset indexes
                  pickle.dump(self.access_order, dataset_pickle_file) # access_order
                  pickle.dump(self.meta, dataset_pickle_file) # meta
                  pickle.dump([self.poses_mean, self.poses_std], dataset_pickle_file) # mean, std
            print('Dataset Saved')
                  
      def _build_meta(self):
            self.meta = np.zeros((self.len, self.nViews, metaDim))
            for i in range(self.len):
                  for j in range(self.nViews):
                        if self.rgb:
                              self.meta[i,j,0] = self.meta_val if self.split == 'train' else -self.meta_val
                        else:
                              self.meta[i,j,0] = self.meta_val if self.split == 'train' else -self.meta_val
                        self.meta[i,j, 1] = i
                        self.meta[i,j, 2] = j
                        self.meta[i, j, 3:5] = np.zeros((2), dtype=np.float32) / 180. * np.arccos(-1)
      
      def _process_subaction(self, subject, action, subaction):
            folder = os.path.join(self.root_dir, subject, self.metadata.action_names[action] + \
                                   '-' + subaction)
            rgb_cameras_folder = os.path.join(folder, self.kFolders['rgb_cameras'])
            tof_folder = os.path.join(folder, self.kFolders['tof_data'])
            
            rgb_folder = os.path.join(rgb_cameras_folder, self.kCameras[0])
            #print('index length: ', len(index))
            self.means = []
            self.std = []
            # Fill in annotations
            annot_file = os.path.join(folder, self.kAnnotationsFilename)
            with h5py.File(annot_file, 'r') as file:
                  frames = file['frame']
                  unique_frames = np.unique(frames)
                  pose2d = file['pose/2d'].value
                  cameras = file['camera'].value
                  index = [{'Views': [], 
                            'Annot': {
                                        'bbox': [], 
                                        '2d': [], 
                                        '3d':[], 
                                        '3d_uncentred': [],
                                        'intrinsic':[], 
                                        'instrinsic-univ':[],
                                        '3d-univ' : [],
                                        '3d-orignial' : [],
                                        }, 
                            'TOF': []}.copy() for i in range(len(unique_frames))]
                  mapping = { f:i for i,f in enumerate(unique_frames) }
                  try:
                        for i,f in enumerate(frames):
                              k = mapping[f]
                              rgb_folder = os.path.join(rgb_cameras_folder, str(cameras[i]))
                              filename = 'img_%06d.jpg' % f
                              tof_filename = 'tof_range%06d.jpg' % f
                              index[k]['Views'] += [os.path.join(rgb_folder, filename)]
                              if len(index[k]['TOF']) == 0:
                                    index[k]['TOF'] = [os.path.join(tof_folder, tof_filename)]
                              index[k]['Annot']['2d'] += [pose2d[i][self.kp_to_use]]
                  except IndexError as e:
                        print(e)
            return index, pose2d[:, self.kp_to_use]
      
      def _build_indexes(self):
            self.dataset_indexes = []
            self.subject_max_idx = []
            self.all_poses = np.asarray([])
            subactions = []
            for subject in self.subjects_to_include:
                  subactions += [ 
                              (subject, action, subaction) 
                              for action, subaction in self.metadata.sequence_mappings[subject].keys() 
                              if int(action) > 1 and 
                              action not in ['54138969', '55011271', '58860488', '60457274']  # Exclude '_ALL' 
                              #                                                                    and Cameras
                              ]
            #print(subactions)
            last_subject, _, _ = subactions[0]
            for subject, action, subaction in tqdm(subactions):
                  if (subject, action, subaction) in self.kBlacklist:
                        continue
                  if last_subject != subject:
                        last_subject = subject
                        self.subject_max_idx += [len(self.dataset_indexes)]
                  indexes, poses = self._process_subaction(subject, action, subaction)
                  self.all_poses = (poses if len(self.all_poses) == 0 \
                                     else np.concatenate((self.all_poses, poses)))
                  self.dataset_indexes += indexes
            self.subject_max_idx += [len(self.dataset_indexes)]
            self.len = len(self.dataset_indexes)
            self._compute_pose_statistics_and_free_poses()
            
      def _compute_pose_statistics_and_free_poses(self):
            self.poses_mean = np.mean(self.all_poses, 0)
            self.poses_std = np.std(self.all_poses, 0)
            del self.all_poses
      
      def _normalize_pose(self, pose):
            return (pose - self.poses_mean) / (self.poses_std + 1e-3)
      
      def _unnormalize_pose(self, pose):
	      return pose *  (torch.from_numpy(self.poses_std).float().to('cpu').unsqueeze(0) + 1e-3) + \
                              torch.tensor(self.poses_mean).float().to('cpu').unsqueeze(0)
      
      def _build_access_index(self):
            self.access_order = []
            last_subject = 0
            for i in self.subject_max_idx:
                  to_use_images = np.arange(last_subject,i,1)[:self.imagesPerSubject]
                  self.access_order += to_use_images.tolist()
                  last_subject = i
            np.random.shuffle(self.access_order)
            self.len = len(self.access_order)
            self.nImages = self.len * self.nViews
            
            
      def shuffle(self):
            self._build_access_index()
      
      def _get_ref(self, idx):
            try:
                return self.dataset_indexes[self.access_order[idx]]
            except IndexError:
                print('trying to access: ' + str(self.access_order[idx]) + 
                       ' out of: ' + str(self.len) + ' || ' + str(len(self.dataset_indexes)))
                print(idx, self.access_order[idx], 'view: ', view)
 
      
      def _load_image(self, idx, view=0):
            image_type = 'Views' if self.rgb else 'TOF'
            filename = self._get_ref(idx)[image_type][view]
            img = Image.open(filename)
            if self.ColorJitter is not None:
                img = self.ColorJitter(img)
            return img
      
      def _load_ref_idx(self, idx_):
            '''
            Get a reference image: ref img is defined as a random frame
            among the same video
            '''
            idx = self.access_order[idx_]

            bin_idx = np.digitize(idx, self.subject_max_idx, right=True)
            low = 0 if bin_idx == 0 else self.subject_max_idx[bin_idx-1]
            high = self.subject_max_idx[bin_idx]
            total_length = high - low
            if total_length < self.ref_interval[0]:
                raise ValueError('Total length shorter then minimum interval')
            lower_interval = [max(idx - self.ref_interval[1], 0), 
                              max(idx - self.ref_interval[0], 0)]
            higher_interval = [min(idx + self.ref_interval[0], high),
                               min(idx + self.ref_interval[1], high)]

            length_lower = lower_interval[1] - lower_interval[0]
            length_higher = higher_interval[1] - higher_interval[0]
            lower_frame = upper_frame = None
            np.random.seed(None)
            if length_lower > 0:
                lower_frame = np.random.randint(lower_interval[0], lower_interval[1])
            if length_higher > 0:
                upper_frame = np.random.randint(higher_interval[0], higher_interval[1])

            if upper_frame is None:
                choose_lower = True
            else:
                choose_lower = (np.random.randint(0,2) > 0 and lower_frame is not None)
            ref_idx = lower_frame if choose_lower else upper_frame

            return self.dataset_indexes[ref_idx]
      
      def _get_normalization_statistics(self):
            return self.poses_mean, (self.poses_std + 1e-7)

      def __getitem__(self, index):
            idx = index % self.len
            imgs = None
            ref_imgs = None
            if not self.gt_only:
                imgs = np.zeros((self.nViews, self.img_size, self.img_size, 3), dtype=np.float32)
                ref_imgs = np.zeros((self.nViews, self.img_size, self.img_size, 3), dtype=np.float32)
            annots = np.zeros((self.nViews, self.K, 2), dtype=np.float32)
            ref_annots = np.zeros((self.nViews, self.K, 2), dtype=np.float32)
            pose_2d = []
            dataset_ref_idx = self._load_ref_idx(idx)
            ratio = float(self.original_size / self.img_size)
            ref_img_type = 'Views' if self.rgb else 'TOF'
            for k in range(self.nViews):
                  if not self.gt_only:
                       if self.rgb:
                            imgs[k] = self.resize_transform(self._load_image(idx, k))            
                            ref_imgs[k] = self.resize_transform(Image.open(dataset_ref_idx[ref_img_type][k]))
                       else:
                            for i in range(3):
                                imgs[k,:,:,i] = self.resize_transform(self._load_image(idx, k))
                                ref_imgs[k,:,:,i] = self.resize_transform(Image.open(dataset_ref_idx[ref_img_type][k]))
                  if self.rgb:
                       annots[k] = self._get_ref(idx)['Annot']['2d'][k]
                       ref_annots[k] = dataset_ref_idx['Annot']['2d'][k]
                  else:
                       annots[k] = self._get_ref(idx)['Annot']['2d'][1]
                       ref_annots[k] = dataset_ref_idx['Annot']['2d'][1]

            inp = None
            if not self.gt_only:
                  imgs = imgs.transpose(0, 3, 1, 2) / 255.
                  ref_imgs = ref_imgs.transpose(0, 3, 1, 2) / 255.
                  inp = torch.from_numpy(imgs)
                  ref_imgs = torch.from_numpy(ref_imgs)
            return inp, annots, ref_imgs, ref_annots 
      
      def __len__(self):
            return self.len
