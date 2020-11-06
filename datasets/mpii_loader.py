import os
import os.path
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as TF
import h5py
from PIL import Image

from tqdm import tqdm
import json_tricks as json

import numpy as np
from scipy.io import loadmat, savemat


def batch_mpii(data):
    def shift_to_interval(x, vis):
        '''
        Make return \in [-1, 1], assuming x \in [0,1]
        '''
        x = x* vis.unsqueeze(-1).type(torch.DoubleTensor)
        if (x.max() > 100):
            print('x > 1: ', x.max())
        if (x.min() < 0):
            print('Error, x < 0', x.min())
        return (2*x - 1)

    def dict_zip(*dcts):
        for i in set(dcts[0]).intersection(*dcts[1:]):
            yield (i,) + tuple(d[i] for d in dcts)

    (imgs, annot) = list(zip(*data))
    def pack(key):
        stack = dict_zip(*annot)
        for x in stack:
            if x[0] == key:
                return torch.from_numpy(np.stack(x[1:],0))
    imgs = torch.from_numpy(np.stack(imgs, 0))

    return {
            "imgs": imgs.type(torch.FloatTensor),
            "annots": pack('joints_3d').type(torch.FloatTensor),
            "kp_mask": pack('joints_3d_vis').unsqueeze(-1).type(torch.FloatTensor),
            }

def LoadMpii(root_dir, train=True, batch_size=16, workers=12):
    img_set = "train" if train else "valid"
    dataset = MPIIDataset(root_dir, img_set=img_set) 
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False, collate_fn=batch_mpii,drop_last=True)
   
    
class MPIIDataset(data.Dataset):
    """
    Class to load PennAction dataset. Assumed folder structure:
    frames/video_number/000x.png
    labels/video_number.mat
    """
    def __init__(self, root_dir, img_set="train"):
        self.root_dir = os.path.expanduser(root_dir)
        self.image_set = img_set
        self.samples = list()
        self.labels = dict()
        self.frame_dir = os.path.join(self.root_dir, "frames")
        self.label_dir = os.path.join(self.root_dir, "labels")
        self.num_joints = 16

        self.samples = self.make_dataset()
        self.samples = self.samples[:200000]

        print('Loading MPIIDataset: %s split: %s' % (self.root_dir, self.image_set))


    def make_dataset(self):
        # create train/val split
        file_name = os.path.join(self.root_dir,
                                 'annot',
                                 self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        samples = []
        counter = {}
        for a in tqdm(anno):
            image_name = a['image']
            if image_name not in counter.keys():
                counter[image_name] = 0

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] 
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images'
            samples.append({
                'image': os.path.join(self.root_dir, image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d[:, 0:2],
                'joints_3d_vis': joints_3d_vis[:, 0],
                'filename': '',
                'imgnum': 0,
                })

        return samples 

    def _load_image(self, path):
         return np.array(Image.open(path))

    def _process_sample(self, img, annot):
        def shift_interval(a):
            return (2*a - 1)
        p_img = img.transpose(2,0,1) / 255.
        annot['joints_3d'] = shift_interval(annot['joints_3d'] / (p_img.shape[1] - 1))
        #annot['joints_3d'] = annot['joints_3d']
        return p_img, annot


    def __getitem__(self, idx):
        idx = idx % self.__len__()
        sample = self.samples[idx]
        img_path = sample['image']
        img_path = img_path.split('/')
        img_path[1] = 'data'
        img_path = '/'.join(img_path)
        img, annot = self._process_sample(self._load_image(img_path), sample)
        return img, annot

    def __len__(self):
        return len(self.samples)

