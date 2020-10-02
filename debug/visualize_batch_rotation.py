import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from datasets.lsp import LoadLsp
import datasets.mpii_loader as mpii
from PIL import Image, ImageDraw
from argparse import ArgumentParser
from modules.util import batch_image_rotation,batch_kp_rotation
import torchvision.transforms as tf
import numpy as np
import cv2
import torch
COLORS = ['black', 'brown', 'royalblue','red','navy', 'orangered','blue', 'tomato', 'purple', 'darkorange', 'darkmagenta', 'darkgoldenrod' , 'cyan', 'yellow', 'white', 'lightgrey']


def tensor_to_image(x):
    out = x.clone().detach().cpu()
    out = out.numpy()
    out = out if out.shape[0] == 3 else np.repeat(out, 3, axis=0)
    out = (out * 255).astype(np.uint8)
    return out

def draw_kp(img_, kps, color='blue'):
    img = img_.transpose(1,2,0) if img_.shape[0] == 3 else img_
    img = Image.fromarray(img)
    kp_img = img.copy()
    draw = ImageDraw.Draw(kp_img)
    radius = 3
    for kp,color in zip(kps,COLORS):
        rect = [kp[0] - radius, kp[1] - radius, kp[0] + radius, kp[1] + radius]
        draw.ellipse(rect, fill=color, outline=color)
    return kp_img

def unnorm_kp(kps):
    return (127./2.) * (kps + 1)

if  __name__ == '__main__':
    dataset = LoadLsp('/hd/levi/dataset/lsp_128')
    indexes = np.random.randint(0,100,5)
    for i in indexes:
        idx = 0
        batch = next(iter(dataset))
        print(batch.keys())
        imgs = batch['imgs']
        rot_annots = batch_kp_rotation(unnorm_kp(batch['annots']), 90)
        annots = batch_kp_rotation(rot_annots, -90)
        img_rots = batch_image_rotation(imgs, 90)
        img_rot_2 = batch_image_rotation(img_rots, -90)
        img = draw_kp(tensor_to_image(img_rot_2[idx]), annots[idx])
        img_rot = draw_kp(tensor_to_image(img_rots[idx]), rot_annots[idx])
        rot2 = imgs[idx]
        #transf = tf.ToPILImage()
        #img = transf(imgs[idx].cpu() * 255.)
        #img_rot = transf(img_rots[idx].cpu() * 255.)
        #rot2 = transf(img_rot_2[idx].cpu() * 255.)
        img.save('img_0.png')
        img_rot.save('img_rot.png')
        #rot2.save('image_rot2.png')



