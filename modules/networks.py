import torch
from torch import nn
import torch.nn.functional as F
from modules.util import EncoderV2, DecoderV2, gaussian2kp
from itertools import product
import numpy as np
from modules.alex_hourglass import Hourglass, HourglassVerbose, make_coordinate_grid, AntiAliasInterpolation2d, DomainAdaptationLayer
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm2d

class KPDetectorConfidence(nn.Module):
    def __init__(self, pretrained_model, num_kp, pad=0, scale_factor=1):

        super(KPDetectorConfidence, self).__init__()
        print(f"n_kp : {num_kp}")
        self.kp_predictor = pretrained_model
        self.heatmap_res = 122
        self.temperature = self.kp_predictor.temperature
        self.scale_factor = self.kp_predictor.scale_factor

        self.sigmoid = torch.nn.Sigmoid()
        self.confidence_conv = nn.Conv2d(in_channels=num_kp, out_channels=1, kernel_size=(2, 2), padding=pad)
        self.confidence_lin = nn.Linear(121*121, 1)


    def gaussian2kp(self, heatmap):
        """
        Extract the mean and the variance from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(heatmap.device)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp


    def forward(self, x):
        with torch.no_grad():
            if self.scale_factor != 1:
                x = self.down(x)

            feature_map = self.kp_predictor.predictor(x)
            prediction = self.kp_predictor.kp(feature_map)

            final_shape = prediction.shape
            heatmap = prediction.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)

            out = self.gaussian2kp(heatmap)
            out['heatmaps'] = heatmap

        out["confidence"] = self.confidence_lin(torch.flatten(self.confidence_conv(prediction), start_dim=1))

        
        return out

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        print(f"n_kp : {num_kp}")
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        
        self.index=0
        self.heatmap_res = 122

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and the variance from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(heatmap.device)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        #print(f"prediction {prediction.shape}")
        #print(f"prediction_flatten {torch.flatten(prediction, start_dim=1).shape}")

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)
        out['heatmaps'] = heatmap

        #out['heatmaps'] = F.sigmoid(prediction)
        #out["confidence"] =  self.confidence_lin(torch.flatten(self.confidence_conv(prediction), start_dim=1))

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        return out

    #def set_domain(self, source=True):
        #self.index = 0 if source else 1

    def convert_bn_to_dial(self,module,device='cuda'):

        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d) or isinstance(child, SynchronizedBatchNorm2d):
                w = child.weight
                b = child.bias
                m = child.running_mean
                v = child.running_var
                #print(w.shape)
                #print(w.mean(),b.mean(),m.mean(),v.mean())
                mod = DomainAdaptationLayer(child.running_mean.shape[0]).to(device)
                #print(mod.weight.mean(),mod.bias.mean(),mod.bn_source.running_mean.mean(),mod.bn_source.running_var.mean(),mod.bn_target.running_mean.mean(),mod.bn_target.running_var.mean())
                # setattr(model, child_name, DomainAdaptationLayer(child.running_mean.shape[0]))
                mod.weight = w.view(1,-1,1,1).to(device)
                mod.bias= b.view(1,-1,1,1).to(device)
                mod.bn_source.running_mean = m
                mod.bn_source.running_var = v
                mod.bn_target.running_mean = m
                mod.bn_target.running_var = v
                setattr(module, child_name, mod)
                #print(mod.weight.mean(),mod.bias.mean(),mod.bn_source.running_mean.mean(),mod.bn_source.running_var.mean(),mod.bn_target.running_mean.mean(),mod.bn_target.running_var.mean())
                #print()
            else:
                self.convert_bn_to_dial(child,device)

    def set_domain(self, m):
        if type(m) == DomainAdaptationLayer:
            m.set_domain(self.index)

    def set_domain_all(self, source):
        self.index = source
        self.apply(self.set_domain)



class KPDetectorVerbose(KPDetector):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetectorVerbose, self).__init__(block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian, scale_factor,
                 single_jacobian_map, pad)
        self.predictor = HourglassVerbose(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map, decoder_feats = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)
        out['features'] = decoder_feats

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        return out



class ImageToSklHourglass(nn.Module):
    """
    Skeleton image generator.
    Takes as input an image and produce its keypoints.
    In the paper is actually implemented as a
    """
    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature, scale_factor=1):
        super(ImageToSklHourglass, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, num_blocks=num_blocks)
        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        heatmap = self.predictor(x)
        final_shape = heatmap.shape
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=3)
        heatmap = heatmap.view(*final_shape)

        return heatmap


class ImageToSkl(nn.Module):
    """
    Skeleton image generator.
    Takes as input an image and produce its keypoints.
    In the paper is actually implemented as a
    """
    def __init__(self,
                 encoder_in_features=3,
                 encoder_channels=32,
                 decoder_channels=256,
                 decoder_out_dim=1,
                 expansion=2,
                 use_sigmoid=True):
        super(ImageToSkl, self).__init__()
        self.encoder = EncoderV2(in_features=encoder_in_features,
                                  channels=encoder_channels,
                                  expansion=expansion)
        decoder_in_features = encoder_channels * (expansion ** 3)
        self.decoder = DecoderV2(in_features=decoder_in_features,
                                  channels=decoder_channels,
                                  output_dim=decoder_out_dim,
                                  reduction=expansion)
        self.use_sigmoid = use_sigmoid
        print('Using ImgToSkl')

    def forward(self, x):
        encoded_img = self.encoder(x)
        generated_img = self.decoder(encoded_img)
        if not self.use_sigmoid:
            return generated_img

        #return F.threshold(torch.sigmoid(generated_img), 1e-2, 0.)
        return torch.sigmoid(generated_img)

class ImageGenerator(nn.Module):
    """
    Skeleton image generator.
    Takes as input an image and produce its keypoints.
    In the paper is actually implemented as a
    """
    def __init__(self,
                 encoder_in_features=3,
                 encoder_channels=32,
                 decoder_channels=256,
                 decoder_out_dim=3,
                 expansion=2,
                 use_sigmoid=True):
        super(ImageGenerator, self).__init__()
        self.encoder = EncoderV2(in_features=encoder_in_features,
                                  channels=encoder_channels,
                                  expansion=expansion)
        decoder_in_features = encoder_channels * (expansion ** 3)
        self.decoder = DecoderV2(in_features=decoder_in_features,
                                  channels=decoder_channels,
                                  output_dim=decoder_out_dim,
                                  reduction=expansion)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        encoded_img = self.encoder(x)
        generated_img = self.decoder(encoded_img)
        if not self.use_sigmoid:
            return generated_img

        return torch.sigmoid(generated_img)



class SklToKP(nn.Module):
    """
    Skeleton to Keypoints converter.
    Given a Skeleton image, extracts its keypoints.
    """
    def __init__(self, channels, n_kp, kp_dim=2, expansion=2, bias=True):
        super(SklToKP, self).__init__()

        self.encoder = EncoderV2(1, channels, expansion=expansion,
                                 last_conv_channels=n_kp)
        in_features=channels*expansion**3
        #self.predictor = nn.Linear(in_features, n_kp*kp_dim, bias=bias)
        self.n_kp = n_kp

    def forward(self, x):
        out = self.encoder(x)
        final_shape = out.shape
        out = out.view(out.shape[0],out.shape[1],-1)
        out = F.softmax(out, dim=-1)
        out = out.view(*final_shape)
        #out = out.mean(dim=-1)
        #out = torch.sigmoid(self.predictor(out).view(out.shape[0], self.n_kp, -1))
        return gaussian2kp(out, None, None)['mean']


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
        self.grid = torch.FloatTensor(target_shape, target_shape, n_coords).to(device)
        grid_ = self.make_coordinate_grid(self.grid.type())

        self.grid = grid_.unsqueeze(2).repeat(1, 1, self.n_edges, 1)
        self.grid = self.grid.unsqueeze(0)

    def make_coordinate_grid(self, type):
            """
            Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
            """
            h = self.target_shape
            w = self.target_shape
            x = torch.arange(w).type(type)
            y = torch.arange(h).type(type)

            x = (2 * (x / (w - 1)) - 1)
            y = (2 * (y / (h - 1)) - 1)

            yy = y.view(-1, 1).repeat(1, w)
            xx = x.view(1, -1).repeat(h, 1)

            meshed = torch.cat([yy.unsqueeze_(2), xx.unsqueeze_(2)], 2)

            return meshed


    def forward(self, kps_, device='cuda'):
        kps = torch.index_select(kps_, 2, torch.LongTensor([1, 0]).to(kps_.device))
        if self.grid.device != kps_.device:
            self.grid = self.grid.to(kps_.device)
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


class ConditionalImageGenerator(nn.Module):
    """
    Conditional Image Generator.
    Given a reference image and a skeleton image, generates an image where
    the object of the reference image has the same pose of the skeleton one.
    keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """
    def __init__(self, in_features=3, hint_features=1, channels=32,
                 expansion=2, decoder_out=256, decoder_out_dim=3):
        super(ConditionalImageGenerator, self).__init__()
        self.encoder_img = EncoderV2(in_features, channels, expansion=expansion)
        self.encoder_skl = EncoderV2(hint_features, channels, expansion=expansion)
        # ********** Had to tweak striding on decoder to make it resize to 224
        decoder_in_features = channels * (expansion ** 3) * 2
        self.decoder = DecoderV2(decoder_in_features, decoder_out,
                                  decoder_out_dim, reduction=expansion)

    def forward(self, reference_img, skl_img):
        encoded_img = self.encoder_img(reference_img)
        encoded_skl = self.encoder_skl(skl_img)
        gen_input = torch.cat([encoded_img,encoded_skl],dim=1)
        generated_img = self.decoder(gen_input)
        return torch.sigmoid(generated_img)

#################################################################################
############################### DISCRIMINATOR ###################################

class DiscriminatorDownBlock2D(nn.Module):
    """
    Downscale block for discriminator
    """
    def __init__(self, in_features, out_features, norm=False, kernel_size=4, stride=2, spectral_norm=False):
        super(DiscriminatorDownBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                               kernel_size=(kernel_size, kernel_size), stride=(stride,stride), padding=1)
        self.norm = None
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        #out = F.avg_pool2d(out, (2,2))
        return out

class Discriminator2D(nn.Module):
    """
    Discriminator for 2D skeletons similar to alex's pix2pix
    """
    def __init__(self, num_channels=1,
                  num_kp=15, kp_variance=0.01,
                  scale_factor=1, block_expansion=64,
                  num_blocks=4, max_features=512, spectral_norm=False):
        super(Discriminator2D, self).__init__()
        #
        # We dont use MovementEmbeddingModule
        #
        down_blocks = []
        self.Interpolation = AntiAliasInterpolation2d(num_channels, scale_factor)
        for i in range(num_blocks):
            down_blocks.append(DiscriminatorDownBlock2D(
                num_channels if i == 0 else \
                min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i+1))),
                norm=(i!=0),
                kernel_size=4,
                stride=(1 if i == num_blocks-1 else 2),
                spectral_norm=spectral_norm))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels,
                               out_channels=1, kernel_size=(4,4), padding=1)

        self.scale_factor = scale_factor

    def forward(self, skl):
        out = skl
        if self.scale_factor != 1:
           out = self.Interpolation(skl)
        #    out = F.interpolate(skl, scale_factor=(self.scale_factor, self.scale_factor), mode='bilinear')
        #out = self.Interpolation(skl)
        for i,down_block in enumerate(self.down_blocks):
            out = down_block(out)
        out = self.conv(out)
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, disc_params, scales=[1]):
        super(MultiScaleDiscriminator, self).__init__()
        models = []
        for x in scales:
            models.append(Discriminator2D(**disc_params, scale_factor=x))
        self.models = nn.ModuleList(models)
        self.scales = scales

    def forward(self, skl):
        scale_maps = []
        for discriminator in self.models:
            scale_maps.append(discriminator(skl))
        return scale_maps


