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
import pickle

def batch_penn(data):
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

    (imgs, kps, vis) = list(zip(*data))
    imgs = torch.from_numpy(np.stack(imgs, 0)).permute(0,3,1,2)
    kps = torch.from_numpy(np.stack(kps, 0)) / (imgs.shape[2]-1)
    vis = torch.from_numpy(np.stack(vis, 0))

    return {
            "imgs": imgs.type(torch.FloatTensor) / 255.,
            "annots": shift_to_interval(kps, vis).type(torch.FloatTensor),
            "kp_mask": vis.unsqueeze(-1).type(torch.FloatTensor),
            }

def LoadPennAction(root_dir, train=True, batch_size=16, workers=12):
   dataset = PennAction(root_dir, train) 
   return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False, collate_fn=batch_penn,drop_last=True)
   
    
class PennAction(data.Dataset):
    """
    Class to load PennAction dataset. Assumed folder structure:
    frames/video_number/000x.png
    labels/video_number.mat
    """
    def __init__(self, root_dir, train=True):
        self.root_dir = os.path.expanduser(root_dir)
        self.train = 1 if train else -1
        self.samples = list()
        self.labels = dict()
        self.frame_dir = os.path.join(self.root_dir, "frames")
        self.label_dir = os.path.join(self.root_dir, "labels")
        self.max_samples = 400000
        self.load_annotations()
        self.make_dataset()
        self.samples = self.samples[:self.max_samples]
        print('Loading PennAction: %s split: %s' % (self.root_dir, ('train' if train else 'test')))


    def make_label(self, video_label, frame_idx):
        x = self.labels[video_label]['x'][frame_idx] 
        y = self.labels[video_label]['y'][frame_idx]
        visibility = self.labels[video_label]['visibility'][frame_idx]
        bbox = None
        try:
            bbox = self.labels[video_label]['bbox'][frame_idx]
        except:
            pass
        kps = np.column_stack((x,y))

        return kps, visibility

    def make_dataset(self):
        for root, _, fnames in tqdm(os.walk(self.frame_dir)):
            video_number = root.split('/')[-1]
            for i, fname in enumerate(sorted(fnames)):
                idx = int(fname.split('.')[0]) - 1
                if self.labels[video_number]['train'][0,0] == self.train:
                    label = self.make_label(video_number, idx) 
                    self.samples.append((os.path.join(root, fname), label))

    def load_annotations(self):
        """
        Function to load annotations in the form:
        dict["video_number"] = {
                                 action: 'tennis_serve',
                                   pose: 'back',
                                      x: [46x13 double]
                                      y: [46x13 double]
                             visibility: [46x13 logical]
                                  train: 1
                                   bbox: [46x4 double]
                             dimensions: [272 481 46]
                                nframes: 46
                               }
        """
        for root, _, fnames in os.walk(self.label_dir):
            for fname in sorted(fnames):
                frame_number = fname.split('.')[0]
                self.labels[frame_number] = scipy.io.loadmat(os.path.join(root, fname))

    def _load_image(self, path):
        return np.array(Image.open(path))

    def __getitem__(self, idx):
        (img_path, (annot, vis)) = self.samples[idx]
        return self._load_image(img_path), annot, vis

    def __len__(self):
        return len(self.samples)

