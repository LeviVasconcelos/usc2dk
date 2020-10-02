import torch
from matplotlib import pyplot as plt
import numpy as np
import time
import torch.nn as nn

def compress_proto(masked, eps=0.000001):
    pooled = masked.sum(-1).sum(-1)
    count = (masked > 0).float().sum(-1).sum(-1)
    count = count + (count == 0.).float() * 1.
    return pooled / (count+eps)


def apply_mask(features, mask):
     return features.unsqueeze(1)*mask.unsqueeze(2)

def apply_mask_pw(features, mask):
    return features*mask


def get_prototype(compressed):
     return compressed.mean(0,keepdim=True)


def adjust_masks(size, masks):
    upsampler = torch.nn.Upsample(size=size, mode='bilinear')
    return upsampler(masks)


class AbstractProtoExtractor(nn.Module):
    """
    Extractor of proto based features
    """
    def __init__(self, edges = None, max_size = (128,128),device='cuda'):
        super(AbstractProtoExtractor, self).__init__()
        rx = []
        ry = []
        for x, y in edges:
            rx.append(x)
            ry.append(y)
        self.edges_x = rx
        self.edges_y = ry
        self.max_size = max_size
        self.device=device
        self.grid = self.make_coordinate_grid()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def make_coordinate_grid(self):
        """
        Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
        """
        h, w = self.max_size
        x = torch.arange(w).float().to(self.device)
        y = torch.arange(h).float().to(self.device)

        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)

        meshed = (torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2).unsqueeze(0).unsqueeze(0) / (h - 1) * 2) - 1

        return meshed

    def pose_masks(self, kp2):
        return None

    def forward(self, feats, masks, compress=False):
        #print(feats,masks.shape)
        adj_masks = adjust_masks((feats.shape[-2],feats.shape[-1]), masks)
        masked_feats = apply_mask(feats,adj_masks)
        x = compress_proto(masked_feats)
        if compress:
            x = get_prototype(x)
        return x

    def forward_apply(self, feats, masks):
        #print(feats,masks.shape)
        adj_masks = adjust_masks((feats.shape[-2],feats.shape[-1]), masks)
        masked_feats = apply_mask_pw(feats,adj_masks)
        return masked_feats



class BoxesProtoExtractor(AbstractProtoExtractor):
    """
    Exctactor of proto based features
    """
    def __init__(self, edges = None, max_size = (128,128), inc=0.1, p=0.1, eps=0.000001, device='cuda'):
        super(BoxesProtoExtractor, self).__init__(edges, max_size, device)
        self.inc = inc
        self.p = p
        self.eps = eps

    def compute_st_distance(self, kp, edges=None):
        edges = self.edges_x
        # TODO
        #if edges is None:
        #    return 1.
        st_distance1 = torch.sum((kp[:,edges[0]] - kp[:,edges[1]]) ** 2)
        st_distance2 = torch.sum((kp[:,edges[2]] - kp[:,edges[3]]) ** 2)
        return torch.sqrt((st_distance1 + st_distance2) / 2.0)

    def make_poly_grid(self, fr, to, st):
        fr = fr + (fr - to) * self.inc
        to = to + (to - fr) * self.inc

        norm_vec = fr - to
        norm_vec = norm_vec.flip(-1)
        norm_vec[:, :, 0] *= -1
        norm = torch.norm(norm_vec, dim=-1).unsqueeze(-1)

        norm_vec = norm_vec / (norm + self.eps)

        v1 = fr + st * self.p * norm_vec
        v2 = fr - st * self.p * norm_vec
        v3 = to - st * self.p * norm_vec
        v4 = to + st * self.p * norm_vec

        a = v1.view(v1.shape[0], v1.shape[1], 1, 1, -1)
        b = v2.view(v1.shape[0], v1.shape[1], 1, 1, -1)
        d = v3.view(v1.shape[0], v1.shape[1], 1, 1, -1)

        am = a - self.grid
        ad = a - d
        ab = a - b

        pmb = (ab * am).sum(-1)
        pmd = (ad * am).sum(-1)
        pab = (ab * ab).sum(-1)
        pad = (ad * ad).sum(-1)

        return ((pmd <= pad) * (pmb <= pab) * (pmb >= 0) * (pmd >= 0)).float()

    def pose_masks(self, kp2):
        st2 = self.compute_st_distance(kp2)
        masks = self.make_poly_grid(kp2[:, self.edges_x], kp2[:, self.edges_y], st2)
        return masks



class BonesProtoExtractor(AbstractProtoExtractor):
    """
    Exctactor of proto based features
    """
    def __init__(self, edges = None, max_size = (128,128), device = 'cuda', gamma=10, n_coords=2):
        super(BonesProtoExtractor, self).__init__(edges, max_size, device)
        self.gamma=gamma
        print('using gamma:', self.gamma)
        self.n_coords = n_coords
        self.n_edges = len(self.edges_x)


    def pose_masks(self, kps):
        #kps = torch.index_select(kps_, 2, torch.LongTensor([1, 0]).to(kps_.device))
        grid_sqz = self.grid.repeat(kps.shape[0], self.n_edges, 1, 1, 1)
        pi_set = kps[:, self.edges_x].view(kps.shape[0], self.n_edges, 1, 1, -1)
        pj_set = kps[:, self.edges_y].view(kps.shape[0], self.n_edges, 1, 1, -1)

        # Compute r
        v_set = (pi_set - pj_set).repeat(1, 1, 128, 128, 1)
        v_norm = v_set.pow(2).sum(-1).unsqueeze(-1)
        u_set = (grid_sqz - pj_set)
        #print( u_set.shape,v_set.shape)
        uv = torch.bmm(u_set.view(-1, 1, self.n_coords), v_set.view(-1, self.n_coords, 1)).view(kps.shape[0],-1,
                                                                                                self.max_size[0], self.max_size[1],
                                                                                                1)

        rs = torch.clamp(uv / v_norm, 0, 1).detach()

        # Compute beta
        betas = torch.exp(-self.gamma * (u_set - rs * v_set).pow(2).sum(-1))
        return betas

class HumansProtoExtractor(BonesProtoExtractor):
    def __init__(self, edges = None, max_size = (128,128), device = 'cuda', gamma=10, n_coords=2):
        super(HumansProtoExtractor, self).__init__(edges, max_size, device, gamma, n_coords)

    def pose_masks(self, kps):
        betas = super(HumansProtoExtractor, self).pose_masks(kps)
        heatmap, _ = betas.max(1)
        return heatmap.unsqueeze(1)


class PartsProtoExtractor(BonesProtoExtractor):
    def __init__(self, edges = None, max_size = (128,128), device = 'cuda', gamma=10, n_coords=2, parts=None):
        super(PartsProtoExtractor, self).__init__(edges, max_size, device, gamma, n_coords)
        self.parts = parts

    def pose_masks(self, kps):
        betas = super(PartsProtoExtractor, self).pose_masks(kps)
        heatmaps = []
        for p in self.parts:
            heatmap, _ = betas[:,p].max(1)
            heatmaps.append(heatmap.unsqueeze(1))
        hm = torch.cat(heatmaps,dim=1)
        #print(hm.shape)
        return hm


class KpsProtoExtractor(AbstractProtoExtractor):
    """
    Exctactor of proto based features
    """
    def __init__(self, edges = None, max_size = (128,128), device = 'cuda', gamma=10):
        super(KpsProtoExtractor, self).__init__(edges, max_size, device)
        self.gamma=gamma


    def pose_masks(self, kps):
        kps = kps.unsqueeze(-2).unsqueeze(-2)
        diff = kps - self.grid

        # Compute beta
        betas = torch.exp(-self.gamma * (diff).pow(2).sum(-1))
        #print(kps)
        return betas



def test_masks(type, device):
    edges = [(0,1), (1,2), (2,3), (0, 4),
                            (4,5), (5,6), (0,7), (7,8),
                            (7,9), (9,10), (10, 11), (7,12),
                            (12, 13), (13, 14)]


    f = type(edges,device=device)

    input = torch.from_numpy(np.load('/home/massimiliano/Downloads/keypoints.npy')).to(device)
    start = time.time()
    out = f.pose_masks(input)
    end = time.time()
    print(end-start)

    i=0
    for i,p in enumerate(out):
        s,_ = torch.max(out[i], 0, out=None)
        plt.imshow(s.to('cpu'))
        plt.show()

    plt.imshow(out[0,0].to('cpu'))
    plt.show()
    plt.imshow(out[0,1].to('cpu'))
    plt.show()
    plt.imshow(out[0,2].to('cpu'))
    plt.show()
