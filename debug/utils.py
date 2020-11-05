import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import numpy as np

from PIL import Image, ImageDraw
COLORS = ['black', 'brown', 'royalblue','red','navy', 'orangered','blue', 'tomato', 'purple', 'darkorange', 'darkmagenta', 'darkgoldenrod' , 'cyan', 'yellow', 'white', 'lightgrey']

def draw_kp(img_, kps, color='blue'):
    img = img_.transpose(1,2,0) if img_.shape[0] == 3 else img_
    img = Image.fromarray(img)
    kp_img = img.copy()
    draw = ImageDraw.Draw(kp_img)
    radius = 3
    print(kps.shape)
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


