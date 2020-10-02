import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt



class KPToSkl(nn.Module):
    """
    Keypoints to Skeleton converter.
    Takes as input a set of keypoints coordinates and generates its skeleton image version.
    """
    def __init__(self, target_shape=128, gamma=0.2, n_coords=2, n_kps=0, edges = None, device='cuda'):
        super(KPToSkl, self).__init__()
        self.gamma = gamma
        self.n_coords = n_coords
        self.valid_edges = edges
        assert len(self.valid_edges)>0

        self.indeces = list(np.arange(n_kps))
        self.target_shape=target_shape
        self.n_kps = n_kps
        self.n_edges = len(edges)

        self.pi_list, self.pj_list = [], []
        for (i, j) in self.valid_edges:
            self.pi_list.append(i)
            self.pj_list.append(j)

        self.grid = torch.FloatTensor(target_shape,target_shape,n_coords).to(device)
        grid_x, grid_y = self.meshgrid(target_shape)
        self.grid[:,:,0] = grid_x
        self.grid[:,:,1] = grid_y
        self.grid/=(target_shape-1)
        self.grid = self.grid.unsqueeze(2).repeat(1, 1, self.n_edges, 1)
        self.grid=self.grid.unsqueeze(0)


    def forward(self, kps, device='cuda'):
        grid_sqz = self.grid.repeat(kps.shape[0], 1, 1, 1, 1)
        pi_set = kps[:, self.pi_list].view(kps.shape[0],1,1,self.n_edges,self.n_coords)
        pj_set = kps[:, self.pj_list].view(kps.shape[0],1,1,self.n_edges,self.n_coords)

        # Compute r
        v_set = (pi_set - pj_set).repeat(1, self.target_shape, self.target_shape, 1, 1)
        v_norm = v_set.pow(2).sum(-1).unsqueeze(-1)
        u_set = (grid_sqz - pj_set)
        uv = torch.bmm(u_set.view(-1,1,self.n_coords),v_set.view(-1, self.n_coords, 1)).view(kps.shape[0], self.target_shape, self.target_shape, -1, 1)
        rs = torch.clamp(uv / v_norm, 0, 1).detach()

        # Compute beta
        betas = torch.exp(-self.gamma * (u_set - rs * v_set).pow(2).sum(-1))


        heatmap, _ = betas.max(-1)
        return heatmap


    def meshgrid(self,n):
        x = torch.arange(n).float()
        grid_x = x[:, None].expand(n, n)
        grid_y = x[None].expand(n, n)
        return grid_x, grid_y

'''edges = [(0,1), (1,2), (2,3), (0, 4),
                        (4,5), (5,6), (0,7), (7,8),
                        (7,9), (9,10), (10, 11), (7,12),
                        (12, 13), (13, 14)]'''


edges = [(0,2), (0,3), (1,2), (1, 3)]



import time

device='cuda'
target_shape = 224
f = KPToSkl(target_shape=target_shape, gamma=200, n_coords=2, n_kps=4, edges = edges, device=device)
input = torch.FloatTensor([[[50,50],[200,200],[50,200],[200,50]]]).to(device)
input = input.repeat(4,1,1)
for i in range(input.shape[0]):
    input[i,:,:]-=(8*i)
input = input/(target_shape-1)

start = time.time()
out = f(input).to('cpu')
end = time.time()
print(end-start)
for i in range(out.shape[0]):
   plt.imshow(out[i,:,:].unsqueeze(-1).repeat(1,1,3))
   plt.show()
