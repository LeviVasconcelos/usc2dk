import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import datasets.penn_action as penn
import datasets.lsp as lsp
import datasets.mpii_loader as mpii
from PIL import Image, ImageDraw
from argparse import ArgumentParser
import numpy as np
import cv2



def draw_kp(img, annots, color="red", bbox=None, center=None):
    kps, vis = annots
    kp_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(kp_img)
    radius = 2
    for i, kp in enumerate(kps):
        color = "red" if not vis[i] else "blue"
        rect = [kp[0] - radius, kp[1] - radius, kp[0] + radius, kp[1] + radius]
        draw.ellipse(rect, fill=color, outline=color)
    if bbox is not None:
        print('bbox: ', bbox)
        draw.rectangle(bbox, outline=color)
    if center is not None:
        rect = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(rect, fill="red", outline="red")
    return np.array(kp_img)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transfrom_img(img, annot):
    s = annot['scale']
    s = s #* np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    r = 0
    image_size = [200, 200]
    c = annot['center']
    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(
        img,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    return input

if  __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", default="/hd/levi/dataset/PennAction/Penn_Action/", help="path to root dir")
    args = parser.parse_args()

    #dataset = penn.PennAction(args.root_dir, [0])
    #dataset = lsp.LSP(args.root_dir)
    dataset = mpii.MPIIDataset(args.root_dir, "train")
    indexes = np.random.randint(0,100,5)
    for i in indexes:
        #img, annots, vis = dataset.__getitem__(i)
        img, annot_dict = dataset.__getitem__(i)
        #bbox = make_box(annot_dict)
        annots = annot_dict['joints_3d']
        vis = annot_dict['joints_3d_vis']
        print(vis)
        print(annots)
        print(annots.shape, vis.shape)

        kp_img = draw_kp(img, (annots, vis))
        Image.fromarray(kp_img).save('mpii_vis %d.png' % i, "PNG")
