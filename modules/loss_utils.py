import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.proto_utils import adjust_masks

def l2_loss(x,y):
    return torch.norm(x-y,dim=1).mean()

#def ce_loss(x,y):
#    return F.cross_entropy(x,y)

def ce_loss(x,y, normalize=False):
    x = x.view(x.shape[0],x.shape[1],-1)
    y = y.view(y.shape[0],y.shape[1],-1)
    #print(x.shape,y.shape)
    if normalize:
        y = y.softmax(1)
    return (-y*F.log_softmax(x,dim=1)).sum(1).mean()
    

def sym_ce_loss(x,y):
    return (ce_loss(x,y,normalize=True)+ce_loss(y,x,normalize=True))/2.


class TemplateDistanceLoss(nn.Module):
    """
    Extractor of proto based features
    """
    def __init__(self, proto_generator, fixed=False, detach=True, protos=None, norm=None, start=0, end=-1):
        super(TemplateDistanceLoss, self).__init__()
        self.proto_generator = proto_generator
        if norm is None:
            norm = nn.MSELoss()
        self.loss = norm
        self.forward = self.forward_protos if fixed else self.forward_feats
        self.protos = protos
        self.detach_mask = detach
        self.start = start
        self.end = end
        print('Set up a Distance Loss with Fixed Protos? ' + str(self.protos is not None) + '    and Masked detached? ' + str(self.detach_mask))

    def forward_feats(self, feats_src, feats_tgt, kps_src, kps_tgt):
        mask_src = self.proto_generator.pose_masks(kps_src).detach()
        mask_tgt = self.proto_generator.pose_masks(kps_tgt)
        #print(mask_src.shape,mask_tgt.shape)
        #exit(1)
        if self.detach_mask:
            mask_tgt = mask_tgt.detach()
        loss = 0
        for (f_src, f_tgt) in zip(feats_src[self.start:self.end],feats_tgt[self.start:self.end]):
            templates_src = self.proto_generator(f_src,mask_src, compress=True)
            templates_tgt = self.proto_generator(f_tgt,mask_tgt, compress=True)
            loss+=self.loss(templates_tgt,templates_src)
        return loss


    def forward_protos(self, feats_src, feats_tgt, kps_src, kps_tgt):
        mask_tgt = self.proto_generator.pose_masks(kps_tgt)
        if self.detach_mask:
            mask_tgt=mask_tgt.detach()

        loss = 0
        for (t_src, f_tgt) in zip(self.protos,feats_tgt[self.start:self.end]):
            templates_tgt = self.proto_generator(f_tgt,mask_tgt, compress=True)
            loss+=self.loss(templates_tgt,t_src)
        return loss


class TemplateTripletLoss(nn.Module):
    """
    Extractor of proto based features
    """
    def __init__(self, proto_generator, detach=True, margin=1.0, p=2, fixed=False, start=0, end=-1):
        super(TemplateTripletLoss, self).__init__()
        self.proto_generator = proto_generator
        self.loss = nn.TripletMarginLoss(margin=margin, p=p)
        self.detach_mask = detach
        self.start = start
        self.end = end
        print('Set up a Triplet Loss with Masked detached? ' + str(self.detach_mask))


    def forward(self, feats_src, feats_tgt, kps_src, kps_tgt):
        mask_src_real = self.proto_generator.pose_masks(kps_src)
        randp = torch.randperm(kps_src.shape[1])
        kps_fake = kps_src[:,randp,:]
        mask_src_fake = self.proto_generator.pose_masks(kps_fake)
        mask_tgt = self.proto_generator.pose_masks(kps_tgt)
        if self.detach_mask:
            mask_tgt =mask_tgt.detach()
        loss = 0
        for (f_src, f_tgt) in zip(feats_src[self.start:self.end],feats_tgt[self.start:self.end]):
            templates_src_real = self.proto_generator(f_src,mask_src_real, compress=True)
            templates_src_fake = self.proto_generator(f_src,mask_src_fake, compress=True)
            templates_tgt = self.proto_generator(f_tgt,mask_tgt, compress=True)
            loss+=self.loss(templates_tgt, templates_src_real.detach(), templates_src_fake.detach())
        return loss


class TemplateMatchingLoss(nn.Module):
    """
    Extractor of proto based features
    """
    def __init__(self, proto_generator, fixed=False, detach=True, protos=None, loss=None, start=0, end=-1):
        super(TemplateMatchingLoss, self).__init__()
        self.proto_generator = proto_generator
        if loss is None:
            loss = l2_loss #sym_ce_loss
        self.loss = loss
        self.detach_mask=detach
        self.forward = self.forward_protos if fixed else self.forward_feats
        self.protos = protos
        self.start = start
        self.end = end
        print('Set up a Matching Loss with Fixed Protos?' + str(self.protos is not None) + '    and Masked detached? ' + str(self.detach_mask))


    def forward_feats(self, feats_src, feats_tgt, kps_src, kps_tgt):
        mask_src = self.proto_generator.pose_masks(kps_src).detach()
        mask_tgt = self.proto_generator.pose_masks(kps_tgt)
        if self.detach_mask:
            mask_tgt=mask_tgt.detach()
        loss = 0
        for (f_src, f_tgt) in zip(feats_src[self.start:self.end],feats_tgt[self.start:self.end]):
            templates_src = self.proto_generator(f_src,mask_src, compress=True).unsqueeze(-1).unsqueeze(-1).squeeze(0)
            similarity_tgt = F.conv2d(f_tgt/f_tgt.norm(dim=1,keepdim=True),(templates_src/templates_src.norm(dim=1,keepdim=True)).detach())
            normed_similarity_tgt = (similarity_tgt+1)/2.
            adj_mask_tgt = adjust_masks((normed_similarity_tgt.shape[-2],normed_similarity_tgt.shape[-1]),mask_tgt)
            loss_c=self.loss(normed_similarity_tgt,adj_mask_tgt)/f_tgt.shape[1]
            loss+=loss_c
        return loss

    def forward_protos(self, feats_src, feats_tgt, kps_src, kps_tgt):
        mask_tgt = self.proto_generator.pose_masks(kps_tgt)
        if self.detach_mask:
            mask_tgt=mask_tgt.detach()
        loss = 0
        for (t_src, f_tgt) in zip(self.protos,feats_tgt[self.start:self.end]):
            t_src = t_src.unsqueeze(-1).unsqueeze(-1).squeeze(0)
            similarity_tgt = F.conv2d(f_tgt/f_tgt.norm(dim=1,keepdim=True),(t_src/t_src.norm(dim=1,keepdim=True)).detach())
            normed_similarity_tgt = (similarity_tgt+1)/2.
            adj_mask_tgt = adjust_masks((normed_similarity_tgt.shape[-2],normed_similarity_tgt.shape[-1]),mask_tgt)
            loss_c=self.loss(normed_similarity_tgt,adj_mask_tgt)/f_tgt.shape[1]
            loss+=loss_c
        return loss

