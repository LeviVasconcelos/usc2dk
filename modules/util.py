from torch import nn

import torch.nn.functional as F
import torch
import torch.nn as nn
import math

import numpy as np
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

class HorizontalMask(nn.Module):
    def __init__(self):
        super(HorizontalMask, self).__init__()
    def forward(self, x):
        coin_flip = torch.LongTensor(1).random_(0, 2)
        mask = torch.ones(x.shape).cuda()
        hf = int((x.shape[-1] - 1)//2)
        if coin_flip > 0.:
            mask[:, :, :, :hf] = 0.
        else:
            mask[:, :, :, hf:] = 0.
        return mask * x


def compute_image_gradient(image, padding=0):
    bs, c, h, w = image.shape

    sobel_x = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])).type(image.type())
    filter = sobel_x.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(image, filter, groups=c, padding=padding)
    grad_x = grad_x

    sobel_y = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).type(image.type())
    filter = sobel_y.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_y = F.conv2d(image, filter, groups=c, padding=padding)
    grad_y = grad_y

    return torch.cat([grad_x, grad_y], dim=1)


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    #adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7
    grid_ = make_coordinate_grid(shape[2:], heatmap.type())
    grid = grid_.unsqueeze(0).unsqueeze(0)

    mean_ = (heatmap * grid)
    mean = mean_.sum(dim=(2, 3))

    kp = {'mean': mean} 

    #if kp_variance == 'matrix':
    #    mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
    #    var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
    #    var = var * heatmap.unsqueeze(-1)
    #    var = var.sum(dim=(3, 4))
    #    var = var.permute(0, 2, 1, 3, 4)
    #    if clip_variance:
    #        min_norm = torch.tensor(clip_variance).type(var.type())
    #        sg = smallest_singular(var).unsqueeze(-1)
    #        var = torch.max(min_norm, sg) * var / sg
    #    kp['var'] = var

    if kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(3, 4))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        var = var.permute(0, 2, 1, 3, 4)
        kp['var'] = var

    return kp

def kp2gaussian2(kp, spatial_size, kp_variance, temp=0.1):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    shape = out.shape
    out = out.view(shape[0], shape[1], -1)
    out = F.softmax(out /  temp, dim=2)
    out = out.view(*shape)
    
    return torch.where(out < 4e-5, torch.zeros(out.shape).cuda(), out)
    #return out


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


class ResBlock3D(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = x
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out



class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock3D(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, temporal=False):
        super(Encoder, self).__init__()

        down_blocks = []

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]

        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False,
                 additional_features_for_block=0, use_last_conv=True):
        super(Decoder, self).__init__()
        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3D((1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (
                2 ** (i + 1))) + additional_features_for_block,
                                       min(max_features, block_expansion * (2 ** i)),
                                       kernel_size=kernel_size, padding=padding))

        self.up_blocks = nn.ModuleList(up_blocks)
        if use_last_conv:
            self.conv = nn.Conv3d(in_channels=block_expansion + in_features + additional_features_for_block,
                                  out_channels=out_features, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = None

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)
        if self.conv is not None:
            return self.conv(out)
        else:
            return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False, ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, temporal=temporal)
        self.decoder = Decoder(block_expansion, in_features, out_features, num_blocks, max_features, temporal=temporal)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv


def matrix_det(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    det = a * d - b * c
    return det


def matrix_trace(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    return a + d


def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm

class UpBlock2D(nn.Module):
    """
    Simple block for processing video (decoder).
    """
    def __init__(self, in_features, out_features, strides=[1,1], kernel_sizes=[3,3], paddings=[1,1], upsample=True):
        super(UpBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=strides[0], kernel_size=kernel_sizes[0],
                              padding=paddings[0])
        self.norm1 = nn.BatchNorm2d(out_features, affine=True)

        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, stride=strides[1], kernel_size=kernel_sizes[1],
                              padding=paddings[1])
        self.norm2 = nn.BatchNorm2d(out_features, affine=True)
        self.upsample=upsample

    def forward(self, x):
        out = x
        if self.upsample:
            out = F.interpolate(x, scale_factor=(2, 2))
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        return out


class DownBlock2D(nn.Module):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self, in_features, out_features, strides=[1,1], kernel_sizes=[3,3], paddings=[1,1]):
        super(DownBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=strides[0], kernel_size=kernel_sizes[0],
                              padding=paddings[0])
        self.norm1 = nn.BatchNorm2d(out_features, affine=True)

        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, stride=strides[1], kernel_size=kernel_sizes[1],
                              padding=paddings[1])
        self.norm2 = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        return out



class EncoderV2(nn.Module):
    """
    Skeleton to Keypoints converter.
    Given a Skeleton image, extracts its keypoints.
    """
    def __init__(self, in_features, channels, expansion=2, last_conv_channels=None):
        super(EncoderV2, self).__init__()

        down_blocks = []
        self.downblock1 = DownBlock2D(in_features, channels, strides=[1,1], kernel_sizes=[7,3], paddings=[3,1])
        in_features = channels
        self.downblock2 = DownBlock2D(in_features, in_features*expansion, strides=[2,1], kernel_sizes=[3,3], paddings=[1,1])
        in_features = in_features * expansion
        self.downblock3 = DownBlock2D(in_features, in_features*expansion, strides=[2,1], kernel_sizes=[3,3], paddings=[1,1])
        in_features = in_features * expansion
        self.downblock4 = DownBlock2D(in_features, in_features*expansion, strides=[2,1], kernel_sizes=[3,3], paddings=[1,1])
        in_features = in_features * expansion
        last_layer_channels = last_conv_channels if last_conv_channels is not None else in_features
        self.final_conv = nn.Conv2d(in_channels=in_features, out_channels=last_layer_channels, stride=1, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.downblock1(x)
        out = self.downblock2(out)
        out = self.downblock3(out)
        out = self.downblock4(out)
        out = self.final_conv(out)
        return out

class DecoderV2(nn.Module):
    """
    Skeleton to Keypoints converter.
    Given a Skeleton image, extracts its keypoints.
    """
    def __init__(self, in_features, channels, output_dim, reduction=2):
        super(DecoderV2, self).__init__()

        down_blocks = []
        self.upblock0 = UpBlock2D(in_features, in_features, strides=[1,1], kernel_sizes=[3,3], upsample=False)
        self.upblock1 = UpBlock2D(in_features, int(in_features/reduction), strides=[1,1], kernel_sizes=[3,3])
        in_features = int(in_features / reduction)
        self.upblock2 = UpBlock2D(in_features, int(in_features/reduction), strides=[1,1], kernel_sizes=[3,3])
        in_features = int(in_features / reduction)
        self.upblock3 = UpBlock2D(in_features, int(in_features/reduction), strides=[1,1], kernel_sizes=[3,3])
        in_features = int(in_features / reduction)
        self.final_conv = nn.Conv2d(in_channels=in_features, out_channels=output_dim, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.upblock0(x)
        out = self.upblock1(out)
        out = self.upblock2(out)
        out = self.upblock3(out)

        out = self.final_conv(out)
        return out

def batch_kp_rotation(kps, angle):
    angle = torch.tensor(angle * math.pi / 180.)
    rot_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                               [-1.*torch.sin(angle), torch.cos(angle)]]).cuda()
    center = kps.mean(1).unsqueeze(1)
    rot_kps = torch.matmul(kps - center, rot_matrix.t())
    return rot_kps + center



def batch_image_rotation(imgs, angle):
    '''
    function to rotate batch of images.
    assumes imgs dimension as: batch x color_channels x width x height
    inspired from: https://github.com/ptrblck/pytorch_misc/blob/master/image_rotation_with_matrix.py
    angles in degrees
    '''

    angle = torch.tensor(angle * math.pi / 180.)
    # Compute 2d rotation matrix
    rot_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                               [-1.*torch.sin(angle), torch.cos(angle)]])

    # Build Mesh grid to be rotated
    xl, yl = torch.meshgrid(torch.arange(imgs.size(2)), torch.arange(imgs.size(3)))
    xl = xl.contiguous()
    yl = yl.contiguous()
    x_mid = (imgs.size(2) + 1) / 2.
    y_mid = (imgs.size(3) + 1) / 2.
    src_ind = torch.cat((
                        (xl.float() - x_mid).view(-1, 1),
                        (yl.float() - y_mid).view(-1, 1)),
                        dim = 1
                        )
    # Rotate indexes with rotation matrix and shift to center
    src_ind = torch.matmul(src_ind, rot_matrix.t())
    src_ind = torch.round(src_ind)
    src_ind += torch.tensor([[x_mid, y_mid]])

    # Set Bounds:
    src_ind[src_ind < 0] = 0.
    src_ind[:, 0][src_ind[:,0] >= imgs.size(2)] = float(imgs.size(2)) - 1
    src_ind[:, 1][src_ind[:,1] >= imgs.size(3)] = float(imgs.size(3)) - 1

    img_out = torch.zeros_like(imgs)
    src_ind = src_ind.long()
    img_out[:, :, xl.view(-1), yl.view(-1)] = imgs[:, :, src_ind[:, 0], \
                                                   src_ind[:, 1]]

    return F.interpolate(img_out, scale_factor=(1,1))

