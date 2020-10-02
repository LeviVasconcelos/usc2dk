import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from datasets.humans36m import LoadHumansDataset
from modules.networks import KPToSkl
from tqdm import tqdm
import yaml
from logger import Visualizer
from PIL import Image, ImageDraw

kFileName = "skeletons_cache"
skeletons = {}
skeletons['humans'] = [(0,1), (1,2), (2,3), (0, 4), 
                        (4,5), (5,6), (0,7), (7,8), 
                        (7,9), (9,10), (10, 11), (7,12), 
                        (12, 13), (13, 14)] 
 
def draw_kp(img, kps):
    print(kps)
    kp_img = img.copy()
    draw = ImageDraw.Draw(kp_img)
    radius = 2
    for kp in kps:
        rect = [kp[0] - radius, kp[1] - radius, kp[0] + radius, kp[1] + radius]
        draw.ellipse(rect, fill="blue", outline="blue")
    return kp_img


def cache_skeletons(log_dir, num_skeletons=20):
    loader = LoadHumansDataset([4], True, num_skeletons, 1, batch_size=num_skeletons) 
    file_name = kFileName
    imgs, skeletons, _ = next(iter(loader))
    torch.save({'images':imgs,
                'skeletons':skeletons,}, os.path.join(log_dir,file_name))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--resample", action='store_true', help="to resample the skeletons")
    parser.add_argument("--log_dir", default=".", help="Folder to store output files")
    parser.add_argument("--n", type=int, default=20, help="number of skeletons to generate")
    parser.add_argument("--config", default="config/u2dkp.yaml", help="indicate config.yaml file")
    opt = parser.parse_args()

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.exists(os.path.join(opt.log_dir, kFileName)) or opt.resample:
        cache_skeletons(opt.log_dir, opt.n)

    with open(opt.config) as f:
        config = yaml.load(f)

    cached_dict = torch.load(os.path.join(opt.log_dir, kFileName)) 
    images = cached_dict['images'].cpu()
    skeletons_kps = cached_dict['skeletons']
    skeletons_kps = skeletons_kps.view(-1, *skeletons_kps.shape[2:])

    ############## keypoint of images[0]
    images = images.view(-1, *images.shape[2:])
    images = (images.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
    original_img = Image.fromarray(images[0])
    kps = skeletons_kps[0].cpu().numpy() 
    kp_img = draw_kp(original_img, kps)
    kp_img.save(os.path.join(opt.log_dir, 'original_kp_image.png'), "PNG")
    original_img.save(os.path.join(opt.log_dir, 'original_image.png'), "PNG")
    
    ############## skeletons[0]
    skeletons_kps /= (128-1)
    config['model_params']['KTS_params']['edges'] = skeletons['humans']
    kp_to_skl = KPToSkl(**config['model_params']['KTS_params'])
    with torch.no_grad():
        skeleton_images = kp_to_skl(skeletons_kps)
    skeleton_image = (skeleton_images.unsqueeze(-1).repeat(1,1,1,3).cpu().numpy())
    skeleton_image = (skeleton_image * 255).astype(np.uint8)
    Image.fromarray(skeleton_image[0]).save(os.path.join(opt.log_dir, 'single_skeleton.png'), "PNG")

    ############### Kp image
    kps_np_img = np.array(kp_img)
    pair_img = np.concatenate([kps_np_img, skeleton_image[0]], axis=1)
    Image.fromarray(pair_img).save(os.path.join(opt.log_dir, 'kp_skel.png'), "PNG")

    ############### Grid
    visualizer = Visualizer()
    image_grid = [] 
    idx = [0 + (i*5) for i in range(0, int(images.shape[0]/5) + 1)]
    for i in range(len(idx)-1):
        image_grid += [skeleton_image[idx[i]:idx[i+1]]]
        image_grid += [images[idx[i]:idx[i+1]]]
    column_img = visualizer.create_image_grid(*image_grid)
    Image.fromarray(column_img).save(os.path.join(opt.log_dir, 'column_img.png'), "PNG")

    
    
