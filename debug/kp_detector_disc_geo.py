import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from datasets.annot_converter import HUMANS_TO_PENN, HUMANS_TO_LSP
from modules.losses import discriminator_gan_loss, generator_loss_2d, generator_gan_loss, masked_l2_loss 
from sync_batchnorm import DataParallelWithCallback
from modules.losses import l2_loss, l1_loss  
from torch.optim.lr_scheduler import MultiStepLR
from modules.util import kp2gaussian2, gaussian2kp
from tqdm import tqdm
import itertools
import yaml
from logger import Visualizer
from evaluation import evaluate
from PIL import Image, ImageDraw
from torchvision.transforms import functional as tf
from modules.vgg19 import Vgg19
import copy
import resource
import random
from modules.util import batch_image_rotation, batch_kp_rotation 

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
COLORS = ['black', 'brown', 'royalblue','red','navy', 'orangered','blue', 'tomato', 'purple', 'darkorange', 'darkmagenta', 'darkgoldenrod' , 'cyan', 'yellow', 'white', 'lightgrey']
def geo_transform(x, d):
    return batch_image_rotation(x, d)

def geo_transform_inverse(x, d):
    return batch_image_rotation(x, d)


class HorizontalMask(nn.Module):
    def __init__(self):
        super(HorizontalMask, self).__init__()
    def forward(self, x):
        coin_flip = torch.LongTensor(1).random_(0, 2)
        mask = torch.ones(x.shape).cuda()
        hf = (x.shape[-1] - 1)//2
        if coin_flip > 0.:
            mask[:, :, :, :hf] = 0.
        else:
            mask[:, :, :, hf:] = 0.
        return mask * x
            
class MaskFromSkeleton(nn.Module):
    def __init__(self):
        super(MaskFromSkeleton, self).__init__()
        self.kernel = torch.ones(3, 3, 3, 3)
        self.times = 3

    def forward(self, x, skeleton):
        self.kernel = self.kernel.to(x.device)
        mask = skeleton.unsqueeze(1).repeat(1, 3, 1, 1)
        for i in range(self.times):
            dilate = torch.nn.functional.conv2d(mask, self.kernel, padding=(1,1))
            mask = torch.clamp(dilate, min=0, max=1)
        return (mask * x).detach()
        
class GeneratorTrainer(nn.Module):
    def __init__(self, generator, discriminator,  
                  kp_to_skl, train_params, conditional_generator=None, 
                  do_inpaint=False, do_recover=False):
        super(GeneratorTrainer, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.conditional_generator = conditional_generator
        self.train_params = train_params
        self.loss_weights = train_params['loss_weights']
        self.mask_layer = HorizontalMask()
        self.mask_skeleton = MaskFromSkeleton()
        self.loss=nn.MSELoss()
        self.vgg = Vgg19().cuda()
        self.kp_to_skl = kp_to_skl
        self.do_inpaint = do_inpaint
        self.do_recover = do_recover

    def perceptual_loss(self, img_gt, img_generated):
        x_vgg = self.vgg(img_generated)
        y_vgg = self.vgg(img_gt)
        total_value = 0
        for i, weight in enumerate(self.loss_weights['perceptual']):
            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            total_value += value * weight

        return total_value

    def inpaint(self, img, hint):
        angle = random.randint(-15,15)
        print('angle: ', angle)
        rot_img = batch_image_rotation(img, angle)
        masked_img = self.mask_layer(rot_img)
        masked_img = batch_image_rotation(masked_img, -angle)
        return self.conditional_generator(masked_img.detach(), hint), masked_img

    def recover_transform(self, heatmaps, tgt_images):
        #kps = gaussian2kp(heatmaps)['mean']
        #skeleton = self.kp_to_skl(kps)
        #masked_images = self.mask_skeleton(tgt_images, skeleton)
        angle = random.randint(-15,15)
        transformed_images = geo_transform(tgt_images, angle) 
        ##### Dropout
        #perm = torch.randperm(heatmaps.shape[1])
        #kNDrop = 7
        ##heatmaps[:, perm[:kNDrop]] = torch.zeros(122,122).cuda()
        #heatmaps[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]] = torch.zeros(122, 122).cuda()
        reconstruction = self.conditional_generator(transformed_images.detach(), heatmaps)
        return {
                "perceptual_loss": self.perceptual_loss(tgt_images, reconstruction),
                "transformed_input": transformed_images,
                "hint": heatmaps,
                "reconstruction": reconstruction,
                }

    def forward(self, images, filter=None, gt=None):
        pred_tgt = self.generator(images)
        kps = pred_tgt['value'] if filter is None else pred_tgt['value'][:, filter]
        generated_skeleton = self.kp_to_skl(kps).unsqueeze(1)
        generated_image = generated_skeleton
        masked_images = images
        #fake_predictions = self.discriminator(generated_skeleton)
        heatmaps = pred_tgt['heatmaps'] if filter is None else pred_tgt['heatmaps'][:, filter]
        if self.do_inpaint:
            generated_image, masked_images = self.inpaint(images, generated_skeleton)
        elif self.do_recover:
            heatmaps_ = kp2gaussian2(gaussian2kp(heatmaps)['mean'], (122, 122), 0.5)
            if gt is not None:
                heatmaps_ = kp2gaussian2(gt, (122,122), 0.5)
            recover_out = self.recover_transform(heatmaps_, images)
            generated_image = recover_out['reconstruction']
            masked_images = recover_out['transformed_input']
            hint = recover_out['hint']
        #generated_skeleton = heatmaps
        fake_predictions = self.discriminator(heatmaps)
        gen_loss = [] 
        for i, map_generated in enumerate(fake_predictions):
            gen_loss.append(generator_gan_loss(discriminator_maps_generated=map_generated,
                                  weight=self.train_params['loss_weights']['generator_gan']).mean())
 
        return {
                "kps": kps,
                "heatmaps": heatmaps,
                "generator_loss": sum(gen_loss)/len(gen_loss),
                "inpaint_loss": 0 if self.do_inpaint is False else self.perceptual_loss(images, generated_image),
                "recover_loss": 0 if self.do_recover is False else recover_out['perceptual_loss'],
                "recover_image": None if self.do_recover is False else recover_out['reconstruction'],
                "image": generated_image,
                "masked_images": masked_images,
                "hint": None if self.do_recover is False else recover_out['hint'],
                }

class DiscriminatorTrainer(nn.Module):
    """
    Skeleton discriminator 
    """
    def __init__(self, discriminators, train_params):
        super(DiscriminatorTrainer, self).__init__()
        self.discriminators = discriminators
        self.train_params = train_params

    def forward(self, gt_image, generated_image):
        gt_maps = self.discriminators(gt_image)
        generated_maps = self.discriminators(generated_image.detach())
        disc_loss = [] 
        for i in range(len(gt_maps)):
            generated_map = generated_maps[i]
            gt_map = gt_maps[i]
            disc_loss.append(discriminator_gan_loss(discriminator_maps_generated=generated_map,
                                     discriminator_maps_real=gt_map,
                                     weight=self.train_params['loss_weights']['discriminator_gan']).mean())

        return { 'loss': (sum(disc_loss) / len(disc_loss)),
                  'scales': disc_loss, }
 
def geo_consistency(y, y_rotated, t, t_inv, d):
    #print('y dim: ', y.shape)
    #print('y_rotated dim: ', y_rotated.shape)
    out = {}
    out['t'] = l1_loss(t(y, d), y_rotated).mean()
    out['t_inv'] = l1_loss(t_inv(y_rotated, -d), y).mean()
    return out

def train_generator_geo(model_generator,
                       discriminator,
                       conditional_generator,
                       model_kp_to_skl,
                       loader_src,
                       loader_tgt,
                       loader_test,
                       train_params,
                       checkpoint,
                       logger, device_ids, 
                       kp_map=None):
    log_params = train_params['log_params']
    optimizer_generator = torch.optim.Adam(model_generator.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    if conditional_generator is not None:
        optimizer_conditional_generator = torch.optim.Adam(conditional_generator.parameters(),
                                                lr=train_params['lr'],
                                                betas=train_params['betas'])
 
    resume_epoch = 0
    resume_iteration = 0
    if checkpoint is not None:
        print('Loading Checkpoint: %s' % checkpoint)
        # TODO: Implement Load/resumo kp_detector
        resume_epoch, resume_iteration = logger.checkpoint.load_checkpoint(checkpoint,
                                              model_generator=model_generator,
                                              optimizer_generator=optimizer_generator,
                                              optimizer_discriminator=optimizer_discriminator,
                                              model_discriminator=discriminator)
        logger.epoch = resume_epoch
        logger.iterations = resume_iteration
    scheduler_generator = MultiStepLR(optimizer_generator, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, 
                                       train_params['epoch_milestones'], 
                                       gamma=0.1, last_epoch=logger.epoch-1)
    if conditional_generator is not None:
        scheduler_conditional_generator = MultiStepLR(optimizer_conditional_generator,
                                           train_params['epoch_milestones'],
                                           gamma=0.1, last_epoch=logger.epoch-1)
    img_generator = GeneratorTrainer(model_generator, 
                                    discriminator, 
                                    model_kp_to_skl, 
                                    train_params, conditional_generator,
                                    do_inpaint=train_params['do_inpaint'],
                                    do_recover=train_params['do_recover']).cuda()

    discriminatorModel = DiscriminatorTrainer(discriminator, train_params).cuda()
    k = 0
    iterator_source = iter(loader_src)
    source_model = copy.deepcopy(model_generator)
    for epoch in range(logger.epoch, train_params['num_epochs']):
        results = evaluate(model_generator, loader_test, train_params['dataset'], kp_map)
        print('Epoch ' + str(epoch)+ ' MSE: ' + str(results))
        logger.add_scalar('MSE test', results['MSE'], epoch)
        logger.add_scalar('PCK test', results['PCK'], epoch)
        if epoch >= 9:
            save_qualy(model_generator, source_model, loader_tgt, epoch, train_params['dataset'], kp_map)
        if epoch > 11:
            return
 
        for i, tgt_batch  in enumerate(tqdm(loader_tgt)):
            try:
                src_batch = next(iterator_source)
            except:
                iterator_source = iter(loader_src)
                src_batch = next(iterator_source)

            angle = random.randint(1,359)
            src_annots = src_batch['annots'].cuda()
            src_images =  kp2gaussian2(src_annots, (122, 122), 0.5)[:, kp_map]
            geo_src_images = kp2gaussian2(batch_kp_rotation(src_annots, angle), (122, 122), 0.5)[:, kp_map]

            #with torch.no_grad():
            #    src_images = model_kp_to_skl(src_annots[:, kp_map]).to('cuda').unsqueeze(1)

            tgt_images = tgt_batch['imgs'].cuda()
            tgt_gt = tgt_batch['annots'].cuda()
            tgt_gt_rot = batch_kp_rotation(tgt_gt, angle)
            geo_tgt_imgs = geo_transform(tgt_images, angle)
            
            pred_tgt = img_generator(tgt_images, kp_map)
            pred_rot_tgt = img_generator(geo_tgt_imgs, kp_map)
            #print('pred_tgt: ', pred_tgt['heatmaps'][0])
            #print('src_annots: ', src_images[0])

            geo_loss = geo_consistency(pred_tgt['heatmaps'], 
                                       pred_rot_tgt['heatmaps'],
                                       geo_transform,
                                       geo_transform_inverse, angle)

            geo_term = geo_loss['t'] + geo_loss['t_inv']
            generator_term = pred_tgt['generator_loss'] + pred_rot_tgt['generator_loss']
            geo_weight = train_params['loss_weights']['geometric']
            recover_loss = 0
            recover_loss = train_params['loss_weights']['recover'] * (pred_tgt['recover_loss'] + pred_rot_tgt['recover_loss'])
            inpaint_loss = 0
            inpaint_loss = train_params['loss_weights']['inpaint'] * pred_tgt['inpaint_loss']
            loss = geo_weight * geo_term + (generator_term) + inpaint_loss + recover_loss
            loss.backward()

            optimizer_generator.step()
            if conditional_generator is not None:
                #print('optimizing inpainting')
                optimizer_conditional_generator.step()
                optimizer_conditional_generator.zero_grad()
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()


            discriminator_no_rot_out = discriminatorModel(gt_image=src_images,
                                                           generated_image=pred_tgt['heatmaps'].detach())
            discriminator_rot_out = discriminatorModel(gt_image=geo_src_images, 
                                                        generated_image=pred_rot_tgt['heatmaps'].detach())

            loss_disc = discriminator_no_rot_out['loss'].mean() + discriminator_rot_out['loss'].mean()
            #if discriminator_no_rot_out['loss'].mean() < 1e-3 and i > 5:
            #print('source min: ', torch.min(src_images[0]))
            #print('source max: ', torch.max(src_images[0]))
            #print('pred min: ', torch.min(pred_tgt['heatmaps'][0]))
            #print('pred max: ', torch.max(pred_tgt['heatmaps'][0]))
 
            #if discriminator_rot_out['loss'].mean() < 1e-3 and i > 5:
            #    print('rotation fucked!!')
            #    print('no rotation fucked!!')
            #    print('source min: ', torch.min(src_images[0]))
            #    print('source max: ', torch.max(src_images[0]))
            #    print('pred min: ', torch.min(pred_rot_tgt['heatmaps'][0]))
            #    print('pred max: ', torch.max(pred_rot_tgt['heatmaps'][0]))
            #    break
            
            loss_disc.backward()
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            if conditional_generator is not None:
                optimizer_conditional_generator.zero_grad()

            logger.add_scalar("Losses", 
                               loss.item(), 
                               logger.iterations)
            logger.add_scalar("Disc Loss", 
                               loss_disc.item(), 
                               logger.iterations)
            logger.add_scalar("Gen Loss", 
                               pred_tgt['generator_loss'].item(), 
                               logger.iterations)
            logger.add_scalar("Geo Loss",
                               (geo_weight * geo_term).item(),
                               logger.iterations)
            logger.add_scalar("Inpaint Loss",
                               inpaint_loss,
                               logger.iterations)
            logger.add_scalar("Recover Loss",
                               recover_loss,
                               logger.iterations)
            scales = discriminator_no_rot_out['scales']
            if len(scales)< 2:
                scales.append(torch.Tensor([0.])) 
                scales.append(torch.Tensor([0.]))
            logger.add_scalar("disc_scales/s1",
                               scales[0].item(),
                               logger.iterations)
            logger.add_scalar("disc_scales/s2",
                               scales[1].item(),
                               logger.iterations)
            logger.add_scalar("disc_scales/s3",
                               scales[2].item(),
                               logger.iterations)
            
            ####### LOG VALIDATION
            if i % log_params['eval_frequency'] == 0 or i==0:
                concat_img = np.concatenate((draw_kp(tensor_to_image(tgt_batch['imgs'][k]), pred_tgt['kps'][k]), 
                                             draw_kp(tensor_to_image(geo_tgt_imgs[k]), pred_rot_tgt['kps'][k])), axis=2)
                #skeletons_img = np.concatenate((draw_kp(tensor_to_image(pred_tgt['image'][k]), pred_tgt['kps'][k]),
                #                                draw_kp(tensor_to_image(pred_rot_tgt['image'][k]), pred_rot_tgt['kps'][k])),
                #                                axis=2)
                skeletons_img = np.concatenate((tensor_to_image(pred_tgt['image'][k]),
                                                tensor_to_image(pred_rot_tgt['image'][k])),
                                                axis=2)
                masked_img = np.concatenate((tensor_to_image(pred_tgt['masked_images'][k]),
                                             tensor_to_image(pred_rot_tgt['masked_images'][k])), axis = 2)
                #heatmap_img_0 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, kp_map[0]].unsqueeze(0), True),
                #                              tensor_to_image(pred_rot_tgt['heatmaps'][k, kp_map[0]].unsqueeze(0), True)), axis=2)
                #heatmap_img_1 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, kp_map[5]].unsqueeze(0), True),
                #                              tensor_to_image(pred_rot_tgt['heatmaps'][k, kp_map[5]].unsqueeze(0), True)), axis=2)
                heatmap_img_0 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, 0].unsqueeze(0), True),
                                              tensor_to_image(pred_rot_tgt['heatmaps'][k, 0].unsqueeze(0), True)), axis=2)
                heatmap_img_1 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, 5].unsqueeze(0), True),
                                              tensor_to_image(pred_rot_tgt['heatmaps'][k, 5].unsqueeze(0), True)), axis=2)
                src_heatmap_0 = np.concatenate((tensor_to_image(src_images[k, 0].unsqueeze(0), True),
                                              tensor_to_image(geo_src_images[k, 0].unsqueeze(0), True)), axis=2)
                src_heatmap_1 = np.concatenate((tensor_to_image(src_images[k, 5].unsqueeze(0), True),
                                              tensor_to_image(geo_src_images[k, 5].unsqueeze(0), True)), axis=2)

               

 
                #inpainted_img = np.concatenate((tensor_to_image(pred_tgt['masked_input'][k]),
                #                              tensor_to_image(pred_tgt['inpainted_img']), axis=2)
                #inpainted_rot_img = np.concatenate((tensor_to_image(pred_rot_tgt['masked_input'][k]),
                #                              tensor_to_image(pred_rot_tgt['inpainted_img']), axis=2)
 
                image = np.concatenate((concat_img, skeletons_img, masked_img), axis=1)
                heatmaps_img = np.concatenate((heatmap_img_0, heatmap_img_1), axis = 1)
                src_heatmaps = np.concatenate((src_heatmap_0, src_heatmap_1), axis = 1)
                #inpaint_vis = np.concatenate((inpainted_img, inpainted_rot_img), axis=1)
                logger.add_image('Pose', image, epoch)
                logger.add_image('heatmaps', heatmaps_img, epoch)
                logger.add_image('src heatmaps', src_heatmaps, epoch)
                #logger.add_image('Reconstruction', inpaint_vis, epoch)
                k += 1
                k = k % len(log_params['log_imgs']) 

            ####### LOG
            logger.step_it()
            
        scheduler_generator.step()
        logger.step_epoch(models = {'model_kp_detector':model_generator,
                                    'optimizer_kp_detector':optimizer_generator})

def draw_kp(img_, kps, color='blue'):
    img = img_.transpose(1,2,0) if img_.shape[0] == 3 else img_
    img = Image.fromarray(img)
    kp_img = img.copy()
    draw = ImageDraw.Draw(kp_img)
    radius = 3
    for kp,color in zip(unnorm_kp(kps),COLORS):
        rect = [kp[0] - radius, kp[1] - radius, kp[0] + radius, kp[1] + radius]
        draw.ellipse(rect, fill=color, outline=color)
    return np.array(kp_img).transpose(2,0,1)

def unnorm_kp(kps):
    return (127./2.) * (kps + 1)

def tensor_to_image(x, heatmap=False):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    if heatmap:
        max_value = np.max(out)
        out = out/max_value 
    out = (out * 255).astype(np.uint8)
    return out

def save_qualy(model, source_model, loader, epoch, dset='mpii', filter=None):
    #save_qualy(model_generator, source_model, loader_test, epoch, kp_map)
    model.eval()
    source_model.eval()
    folder = 'images_clean_%s/%d' % (dset, epoch)
    if not os.path.exists('images_clean_%s' % dset):
        os.mkdir('images_clean_%s' % dset)
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('saving on: ', folder)
    with torch.no_grad():
        for i,batch in tqdm(enumerate(loader)):
            imgs_cuda = batch['imgs'].to('cuda')
            out = model(imgs_cuda)
            src_out = source_model(imgs_cuda)

            for k in range(imgs_cuda.shape[0]):
                model_img = draw_kp(tensor_to_image(imgs_cuda[k]), out['value'][k, filter])
                source_img = draw_kp(tensor_to_image(imgs_cuda[k]), src_out['value'][k, filter])
                clear_img = tensor_to_image(imgs_cuda[k])
                Image.fromarray(model_img.transpose(1,2,0)).save(os.path.join(folder, '%d_%d_codel.png' % (i,k)), "PNG")
                Image.fromarray(source_img.transpose(1,2,0)).save(os.path.join(folder, '%d_%d_src.png' % (i,k)), "PNG")
                Image.fromarray(source_img.transpose(1,2,0)).save(os.path.join(folder, '%d_%d_clear.png' % (i,k)), "PNG")
    model.train()


