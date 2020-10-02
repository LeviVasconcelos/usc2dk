import torch.utils.data as data
import torch
import numpy as np
from datasets.humans36m import Humans36mDataset

def flatten_views(img):
    return img.view(-1, *img.shape[2:])

class Batch:
    def __init__(self, data):
        (img, annot, ref_img, unaligned_annot) = list(zip(*data))
        self.imgs = flatten_views(torch.stack(img, 0))
        self.ref_imgs = flatten_views(torch.stack(ref_img, 0))
        self.annots = flatten_views(torch.from_numpy(np.stack(annot, 0))) / (self.imgs.shape[2]-1)
        self.unaligned_annots = flatten_views(torch.from_numpy(np.stack(unaligned_annot, 0))) / (self.imgs.shape[2]-1)


def process_batch(data):
    def shift_to_interval(x):
        '''
        Make return \in [-1, 1], assuming x \in [0,1]
        '''
        return (2*x - 1)

    ziped_data = list(zip(*data))
    #len(ziped_data)
    #print(ziped_data[0])
    (img, annot, ref_img, ref_annot, unaligned_annot) = list(zip(*data))
    imgs = flatten_views(torch.stack(img, 0))
    ref_imgs = flatten_views(torch.stack(ref_img, 0))
    annots = flatten_views(torch.from_numpy(np.stack(annot, 0))) / (imgs.shape[2]-1)
    ref_annots = flatten_views(torch.from_numpy(np.stack(ref_annot, 0))) / (imgs.shape[2]-1)
    unaligned_annots = flatten_views(torch.from_numpy(np.stack(unaligned_annot, 0))) / (imgs.shape[2]-1)

    out = {
            "imgs":imgs,
            "ref_imgs":ref_imgs,
            "ref_annots":shift_to_interval(ref_annots),
            "annots":shift_to_interval(annots),
            "unaligned_annots":shift_to_interval(unaligned_annots)
            }
    return out 

def process_batch_class(data):
    return Batch(data)

def LoadUnalignedH36m(subjects_data, subjects_gt, rgb,
                      nImages, nViews, batch_size=1, img_size=128, workers=4, ref_interval=[3,30]):
    data_h36m = Humans36mDataset(nViews, '', rgb, 
                                   nImages, subjects_data, -1, 
                                   img_size=img_size, normalized=False, ref_interval=ref_interval)
    gt_h36m =  Humans36mDataset(nViews, '', rgb, 
                                   nImages, subjects_gt, -1, 
                                   img_size=img_size, normalized=False, gt_only=True)
    print('Len data: %d' % len(data_h36m))
    print('Len unaligned: %d' % len(gt_h36m))
    dset = UnalignedH36m(data_h36m, gt_h36m) 
    loader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, 
                              num_workers=workers, pin_memory=False, collate_fn=process_batch)
    #loader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, 
    #                          num_workers=workers, pin_memory=False)
 
    return loader

class UnalignedH36m(data.Dataset):
    """
    Class unifying the loading process of data and unaligned gt
    """
    def __init__(self, data_sampler, gt_sampler):
        self.data_sampler = data_sampler
        self.gt_sampler = gt_sampler

    def shuffle_unaligned(self):
        print('shuffling unaligned\n')
        self.gt_sampler.shuffle()

    def __getitem__(self, i):
        j = i % len(self.data_sampler)
        k = i % len(self.gt_sampler)
        img, annots, ref_img, ref_annots = self.data_sampler.__getitem__(j)
        _, unalign_gt, _, _ = self.gt_sampler.__getitem__(k)
        return (img, annots, ref_img, ref_annots, unalign_gt)

    def __len__(self):
        return len(self.data_sampler)


