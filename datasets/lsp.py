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
from torchvision.transforms import ColorJitter, ToTensor, ToPILImage

from tqdm import tqdm
import pickle

def batch_lsp(data):
    def shift_to_interval(x, vis):
        '''
        Make return \in [-1, 1], assuming x \in [0,1]
        '''
        x = x* vis.unsqueeze(-1).type(torch.DoubleTensor)
        if (x.max() > 10):
            print('x > 1: ', x.max())
        if (x.min() < 0):
            print('Error, x < 0', x.min())
        return (2*x - 1)

    (imgs, jit_imgs, kps, vis) = list(zip(*data))
    [np(['levi']), [np('ricardo')]]
    imgs = torch.from_numpy(np.stack(imgs, 0)).permute(0,3,1,2)
    jit_imgs = torch.from_numpy(np.stack(jit_imgs, 0)).permute(0,3,1,2)
    kps = torch.from_numpy(np.stack(kps, 0)) / (imgs.shape[2]-1)
    vis = torch.from_numpy(np.stack(vis, 0))


    return {
            "imgs": imgs.type(torch.FloatTensor) / 255.,
            "jit_imgs": jit_imgs.type(torch.FloatTensor) / 255.,
            "annots": shift_to_interval(kps, vis).type(torch.FloatTensor),
            "annots_unnormed": kps * (imgs.shape[2]-1),
            "kp_mask": vis.unsqueeze(-1).type(torch.FloatTensor),
            }

def LoadLsp(root_dir, train=True, batch_size=16, workers=12, use_jitter=False):
   dataset = LSP(root_dir, train, use_jitter) 
   return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=workers, 
                                      pin_memory=False, drop_last=True, collate_fn=batch_lsp)
   
    
class LSP(data.Dataset):
    """
    Class to load LSP dataset. Assumed folder structure:
    root/images/im000x.jpg
    root/joints.mat
    """
    def __init__(self, root_dir, train=True, use_jitter=False):
        self.root_dir = os.path.expanduser(root_dir)
        self.train = train
        self.keypoints = None
        self.visibility = None
        self.samples = list()
        self.use_jitter = use_jitter
        self.ColorJitter = None
        if self.use_jitter:
            self.ColorJitter = ColorJitter(brightness = 0,
                                            contrast=0,
                                            saturation=0,
                                            hue=0.5)

        self.frames_dir = os.path.join(self.root_dir, "images")

        self.load_annotations()
        print('keypoints: ', self.keypoints.shape)
        print('visibility: ', self.visibility.shape)
        self.make_dataset()
        print('Length: ', len(self.samples))
        print('Loading LSP: %s split: %s' % (self.root_dir, ('train' if self.train else 'test')))


    def make_label(self, image_name, idx):
        return (os.path.join(self.frames_dir, image_name), (self.keypoints[idx], self.visibility[idx]))

    def make_dataset(self):
        for root, subs, fnames in tqdm(os.walk(self.frames_dir)):
            for i, fname in enumerate(sorted(fnames)):
                if i >= 1000 and self.train == True:
                    return
                if i < 1000 and self.train == False:
                    continue
                self.samples.append(self.make_label(fname, i))


    def load_annotations(self):
        """
        Function to load annotations in the form:
        numpy[frame_number] = {
                                    kps: [14x2]
                                kp_mask: [14x1 logical]
                               }
        """
        filename = os.path.join(self.root_dir, 'joints.mat')
        try:
            annots = scipy.io.loadmat(filename)
        except:
            raise ValueError('Could not read file %s' % filename)

        self.keypoints = annots['joints'][:2].transpose(2,1,0)
        self.visibility = annots['joints'][2].transpose(1,0) 

    def _load_image(self, path):
        return Image.open(path)

    def __getitem__(self, idx):
        (img_path, (annot, vis)) = self.samples[idx]
        img = self._load_image(img_path)
        annot['path'] = img_path
        jit_img = img
        if self.use_jitter:
            jit_img = self.ColorJitter(img)

        return np.array(img), np.array(jit_img), annot, vis

    def __len__(self):
        return len(self.samples)

