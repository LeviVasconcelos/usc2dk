import os
import sys
import itertools
import copy
import resource
import random
import yaml
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as tf
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from modules.losses import discriminator_gan_loss, generator_loss_2d
from modules.losses import generator_gan_loss, masked_l2_loss
from modules.losses import l2_loss, l1_loss  
from modules.util import kp2gaussian2, gaussian2kp
from modules.util import batch_image_rotation, batch_kp_rotation 
from modules.vgg19 import Vgg19

from datasets.annot_converter import HUMANS_TO_PENN, HUMANS_TO_LSP
from sync_batchnorm import DataParallelWithCallback
from logger import Visualizer
from evaluation import evaluate  

from modules.affine_augmentation import batch_kp_affine, batch_img_affine, inverse_aff_values, inverse_affine, batch_affine


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
COLORS = ['black', 'brown', 'royalblue','red','navy', 'orangered','blue', 'tomato', 'purple', 'darkorange', 'darkmagenta', 'darkgoldenrod' , 'cyan', 'yellow', 'white', 'lightgrey']

def geo_transform(x, d):
    return batch_image_rotation(x, d)

def geo_transform_inverse(x, d):
    return batch_image_rotation(x, d)
    
def d_unrolled_loop(d_gen_input=None, real_data=None,optimizer_discriminator=None, gen=None,kp_map=None, discriminator=None, train_params=None):
    
    optimizer_discriminator.zero_grad()
    with torch.no_grad():
        fake_data = gen(d_gen_input,kp_map)
    
    gt_maps = discriminator(real_data)
    generated_maps = discriminator(fake_data['heatmaps'].detach())
    disc_loss = [] 
    for i in range(len(gt_maps)):
        generated_map = generated_maps[i]
        gt_map = gt_maps[i]
        disc_loss.append(discriminator_gan_loss(discriminator_maps_generated=generated_map,
                                    discriminator_maps_real=gt_map,
                                    weight=train_params['loss_weights']['discriminator_gan']).mean())
    out_disc = { 'loss': (sum(disc_loss) / len(disc_loss)),
                  'scales': disc_loss, }

    loss_disc = out_disc['loss'].mean() 
    loss_disc.backward(create_graph=True)


    optimizer_discriminator.step()
    optimizer_discriminator.zero_grad()

class Buffer():
    def __init__(self, buff_dim=200):
        self.buff_dim = buff_dim
        self.sampling_dim = None
        self.buffer_no_rot = torch.Tensor([]).cuda()
        self.buffer_rot = torch.Tensor([]).cuda()
    
    def put_hm(self, buffer, hm):
        # randomly select half of the samples from the heatmaps
        if self.sampling_dim is None:
            self.sampling_dim =int(hm.shape[0]/2)
            print(f"sampling dim setted at {self.sampling_dim}")
        random_samples = list(range(0,int(hm.shape[0]/2))) 
        random.shuffle(random_samples)
        random_samples = random_samples[:int(hm.shape[0]/2)]
        hm = hm[random_samples]
        
        if buffer.shape[0]+hm.shape[0] <= self.buff_dim:
            buffer = torch.cat((buffer,hm),0)
        else:
            # compute the overflow if the new hm are added 
            overflow = buffer.shape[0]+hm.shape[0] - self.buff_dim
            # sample randomly the right amount of sample to have the buffer always filled
            random_samples = list(range(0,buffer.shape[0])) 
            random.shuffle(random_samples)
            random_samples = random_samples[:(buffer.shape[0]-overflow)]
            # remove some randome elemnts
            buffer = buffer[random_samples]
            # sample from it
            buffer = torch.cat((buffer,hm),0)
        return buffer 

    def put_no_rot(self, no_rot_hm):
        self.buffer_no_rot = self.put_hm(self.buffer_no_rot,no_rot_hm)
    def put_rot(self, rot_hm):
        self.buffer_rot = self.put_hm(self.buffer_rot,rot_hm)
        
    def get_samples(self, buffer):
        random_samples = list(range(0,buffer.shape[0]))
        random.shuffle(random_samples)
        random_samples = random_samples[:self.sampling_dim]
        return buffer[random_samples]
    
    def get_no_rot(self):
        return self.get_samples(self.buffer_no_rot)
    def get_rot(self):
        return self.get_samples(self.buffer_rot)
            
class GeneratorTrainer(nn.Module):
    def __init__(self, generator, discriminator,  
                  train_params):
        super(GeneratorTrainer, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.loss_weights = train_params['loss_weights']
        self.loss=nn.MSELoss()
        self.vgg = Vgg19().cuda()

    def perceptual_loss(self, img_gt, img_generated):
        x_vgg = self.vgg(img_generated)
        y_vgg = self.vgg(img_gt)
        total_value = 0
        for i, weight in enumerate(self.loss_weights['perceptual']):
            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
            total_value += value * weight
        return total_value

    def forward(self, images, filter=None, gt=None):
        pred_tgt = self.generator(images)
        kps = pred_tgt['value'] if filter is None else pred_tgt['value'][:, filter]
        masked_images = images
        heatmaps = pred_tgt['heatmaps'] if filter is None else pred_tgt['heatmaps'][:, filter]
        fake_predictions = self.discriminator(heatmaps)
        gen_loss = [] 
        for i, map_generated in enumerate(fake_predictions):
            gen_loss.append(generator_gan_loss(discriminator_maps_generated=map_generated,
                                  weight=self.train_params['loss_weights']['generator_gan']).mean())
 
        return {
                "kps": kps,
                "heatmaps": heatmaps,
                "generator_loss": sum(gen_loss)/len(gen_loss),
                }

class DiscriminatorTrainer(nn.Module):
    """
    Skeleton discriminator 
    """
    def __init__(self, discriminators, train_params):
        super(DiscriminatorTrainer, self).__init__()
        self.discriminators = discriminators
        self.train_params = train_params

    def forward(self, gt_image, generated_image, filter=None):
        
        #gt_image[""]
        #print(f"gt_image {gt_image.shape}")
        #print(f"generated_image {generated_image.shape}")
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
    out = {}
    #print(f"y_rot {y_rotated.shape}")
    #print(f"y {y.shape}")
    out['t'] = l1_loss(t(y, d), y_rotated).mean()
    out['t_inv'] = l1_loss(t_inv(y_rotated, -d), y).mean()
    return out

def geo_consistency_affine(y, y_affine,aff_matrix):
    out = {}
    #print(f"y_rot {y_rotated.shape}")
    #print(f"y {y.shape}")
    #[batchsize, num_kp, 122, 122]
    # apply the affine transf to the non rotated heatmpas
    orig_to_aff_hm = batch_affine(y,aff_matrix, inverse=False)
    out['t'] = l1_loss(orig_to_aff_hm, y_affine).mean()

    # apply the inverse affine trans to the rotated heatmaps
    inverse_aff = inverse_aff_values(aff_matrix)
    aff_to_orig_hm = batch_affine(y_affine,inverse_aff, inverse=True)

    out['t_inv'] = l1_loss(aff_to_orig_hm, y).mean()
    return out

### compute the angle difference between the kps
def angle_difference(kps,rot_kps,gt_angle,loss, device="cuda"):
    angle = torch.atan2(torch.det(torch.stack([rot_kps,kps], 2)),torch.sum(rot_kps*kps, dim=2))
    gt_angle = torch.tensor(gt_angle).type(torch.FloatTensor)
    gt_angle = gt_angle.cuda()
    angle = (angle*180/np.pi).mean()
    return loss(angle,gt_angle)

def train_generator_geo(model_generator,
                       discriminator,
                       loader_src,
                       loader_tgt,
                       loader_test,
                       train_params,
                       checkpoint,
                       logger, 
                       device_ids,
                       kp_map=None):
    log_params = train_params['log_params']
    optimizer_generator = torch.optim.Adam(model_generator.parameters(),
                                            lr=train_params['lr'],
                                            betas=train_params['betas'])
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
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
    img_generator = GeneratorTrainer(model_generator, 
                                    discriminator, 
                                    train_params).cuda()
    img_generator = DataParallelWithCallback(img_generator, device_ids=device_ids)

    discriminatorModel = DiscriminatorTrainer(discriminator, train_params).cuda()
    discriminatorModel = DataParallelWithCallback(discriminatorModel, device_ids=device_ids)
    buffer = Buffer()
    k = 0
    kpVariance = train_params['heatmap_var']
    iterator_source = iter(loader_src)
    source_model = copy.deepcopy(model_generator)
    
    unrolled_steps = train_params["unrolled_steps"] 
    buffer_flag = train_params["buffer"] 
    affine_augmentation = train_params["affine_augmentation"]
    # save the model if a new max pck is reached
    max_pck = None
    geom_attention = train_params["geom_attention"]

    reload_disc = train_params["reload_disc"]

    reload_buffer = 10
    if reload_disc:
        old_disc = copy.deepcopy(discriminator.state_dict())


    angle_difference_flag = train_params["angle_difference"] # use the angle difference instead of equivariance
    geo_loss = torch.nn.MSELoss()
    print(f"buffer: {buffer_flag} unrolled_steps: {unrolled_steps} affine: {affine_augmentation} angle_diff_equivariance:  {angle_difference_flag}")


    for epoch in range(logger.epoch, train_params['num_epochs']):
        results = evaluate(model_generator, loader_test, train_params['dataset'], kp_map)
        print('Epoch ' + str(epoch)+ ' MSE: ' + str(results))
        logger.add_scalar('MSE test', results['MSE'], epoch)
        logger.add_scalar('PCK test', results['PCK'], epoch)

        if max_pck != None and results["PCK"].numpy() > max_pck:
            max_pck = results["PCK"].numpy()
            logger.save_model(models = {'model_kp_detector':model_generator,
                                    'optimizer_kp_detector':optimizer_generator})
        elif max_pck == None :
            max_pck = results["PCK"].numpy()
            
        for i, tgt_batch  in enumerate(tqdm(loader_tgt)):            
            try:
                src_batch = next(iterator_source)
            except:
                iterator_source = iter(loader_src)
                src_batch = next(iterator_source)

            src_annots = src_batch['annots'].cuda()
            src_images =  kp2gaussian2(src_annots, (122, 122), kpVariance).detach()   
            tgt_images = tgt_batch['imgs'].cuda()
            tgt_gt = tgt_batch['annots'].cuda()

            if not affine_augmentation:
                range_angle = int( train_params["angle_range"] * (epoch/train_params['num_epochs']))
                angle = random.randint(-1*range_angle,range_angle)
                geo_src_images = kp2gaussian2(batch_kp_rotation(src_annots, angle), (122, 122), kpVariance).detach()
                tgt_gt_rot = batch_kp_rotation(tgt_gt, angle)
                tgt_gt_rot = tgt_gt_rot.detach()
                geo_tgt_imgs = geo_transform(tgt_images, angle).detach()
            else:
                # perform affine transformation
                geo_tgt_imgs, aff_matrix = batch_img_affine(tgt_images) # uses the predefined angle translation and ranges
                geo_tgt_imgs = geo_tgt_imgs.detach()
                tgt_gt_rot = batch_kp_affine(tgt_gt,aff_matrix)
                tgt_gt_rot = tgt_gt_rot.detach()
                src_annots = batch_kp_affine(tgt_gt,aff_matrix)
                src_annots = src_annots.detach()
                geo_src_images = kp2gaussian2(src_annots, (122, 122), kpVariance)
                geo_src_images = geo_src_images.detach()

            if reload_disc and reload_buffer % epoch ==0:
                reload_disc = copy.deepcopy(old_disc)
                old_disc = copy.deepcopy(discriminator.state_dict())
                discriminator.load_state_dict(reload_disc)


            ## code adapted from https://github.com/andrewliao11/unrolled-gans
            ## Unroll the discriminator here making a deep copy
            if unrolled_steps > 0:
                backup = copy.deepcopy(discriminator.state_dict())
                for iterat in range(unrolled_steps):
                    # unrolled loop for non rotated frames
                    d_unrolled_loop(d_gen_input=tgt_images, real_data=src_images,optimizer_discriminator=optimizer_discriminator, gen=img_generator, kp_map=kp_map, discriminator=discriminator, train_params=train_params)
                    # unrolled lopp for rot frames
                    if geom_attention == 1:
                        d_unrolled_loop(d_gen_input=geo_tgt_imgs, real_data=geo_src_images,optimizer_discriminator=optimizer_discriminator, gen=img_generator, kp_map=kp_map, discriminator=discriminator, train_params=train_params)
                
            pred_tgt = img_generator(tgt_images, kp_map) 
            pred_rot_tgt = img_generator(geo_tgt_imgs, kp_map) 
            # compute the equivariance using the angle difference
            if angle_difference_flag:
                geo_term = angle_difference(pred_tgt["kps"], pred_rot_tgt["kps"], angle, geo_loss)
            else:
                if not affine_augmentation:
                    geo_loss = geo_consistency(pred_tgt['heatmaps'], 
                                            pred_rot_tgt['heatmaps'],
                                            geo_transform,
                                            geo_transform_inverse, angle)
                else:
                    geo_loss = geo_consistency_affine(pred_tgt['heatmaps'], 
                                            pred_rot_tgt['heatmaps'],
                                            aff_matrix)

                geo_term = geo_loss['t'] + geo_loss['t_inv']
            generator_term = pred_tgt['generator_loss'] + geom_attention * pred_rot_tgt['generator_loss'] 
            geo_weight = train_params['loss_weights']['geometric']
            loss = geo_weight * geo_term + (generator_term) 
            loss.backward()

            optimizer_generator.step()
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            # reload disc model
            if unrolled_steps > 0:
                discriminator.load_state_dict(backup)
            if buffer_flag:
                #### save new frame in the buuffer
                buffer.put_no_rot(pred_tgt['heatmaps'].detach())
                buffer.put_rot(pred_rot_tgt['heatmaps'].detach())
                #### load frame from the buffer 
                random_samples = list(range(0,pred_rot_tgt['heatmaps'].shape[0]))
                random.shuffle(random_samples)
                random_samples = random_samples[:buffer.sampling_dim]
                gen_hm_no_rot = torch.cat((buffer.get_no_rot(),pred_tgt['heatmaps'][random_samples].detach()), 0)
                gen_hm_rot = torch.cat((buffer.get_rot(),pred_rot_tgt['heatmaps'][random_samples].detach()), 0)
                discriminator_no_rot_out = discriminatorModel(gt_image=src_images,
                                                            generated_image=gen_hm_no_rot)
                discriminator_rot_out = discriminatorModel(gt_image=geo_src_images, 
                                                            generated_image=gen_hm_rot)
            else:
                discriminator_no_rot_out = discriminatorModel(gt_image=src_images,
                                                            generated_image=pred_tgt['heatmaps'].detach())
                discriminator_rot_out = discriminatorModel(gt_image=geo_src_images, 
                                                            generated_image=pred_rot_tgt['heatmaps'].detach())

            loss_disc = discriminator_no_rot_out['loss'].mean() + geom_attention * discriminator_rot_out['loss'].mean()
            loss_disc.backward()

            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()

            logger.add_scalar("Losses", 
                               loss.item(), 
                               logger.iterations)
            logger.add_scalar("Disc Loss", 
                                loss_disc.item(), 
                                logger.iterations)
            logger.add_scalar("Gen Loss", 
                               pred_tgt['generator_loss'].item(), 
                               logger.iterations)
            logger.add_scalar("Weighted Geo Loss",
                               (geo_weight * geo_term).item(),
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
                heatmap_img_0 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, 0].unsqueeze(0), True),
                                              tensor_to_image(pred_rot_tgt['heatmaps'][k, 0].unsqueeze(0), True)), axis=2)
                heatmap_img_1 = np.concatenate((tensor_to_image(pred_tgt['heatmaps'][k, 5].unsqueeze(0), True),
                                              tensor_to_image(pred_rot_tgt['heatmaps'][k, 5].unsqueeze(0), True)), axis=2)
                src_heatmap_0 = np.concatenate((tensor_to_image(src_images[k, 0].unsqueeze(0), True),
                                              tensor_to_image(geo_src_images[k, 0].unsqueeze(0), True)), axis=2)
                src_heatmap_1 = np.concatenate((tensor_to_image(src_images[k, 5].unsqueeze(0), True),
                                              tensor_to_image(geo_src_images[k, 5].unsqueeze(0), True)), axis=2)

               
                image = concat_img
                heatmaps_img = np.concatenate((heatmap_img_0, heatmap_img_1), axis = 1)
                src_heatmaps = np.concatenate((src_heatmap_0, src_heatmap_1), axis = 1)
                logger.add_image('Pose', image, epoch)
                logger.add_image('heatmaps', heatmaps_img, epoch)
                logger.add_image('src heatmaps', src_heatmaps, epoch)
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


