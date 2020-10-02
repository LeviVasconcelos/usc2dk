import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy

from modules.networks import KPToSkl, SklToKP 
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.lsp import LoadLsp
from datasets.h36m_skeletons import LoadSkeletonDataset
from datasets.annot_converter import HUMANS_TO_MPII, HUMANS_TO_LSP, HUMANS_TO_PENN
from debug.test_skeleton_to_keypoints import debug_encoder

skeletons = {}
skeletons['humans'] = [(0,1), (1,2), (2,3), (0, 4), 
                        (4,5), (5,6), (0,7), (7,8), 
                        (7,9), (9,10), (10, 11), (7,12), 
                        (12, 13), (13, 14)] 
skeletons['penn'] = [(0,1), (0,2), (1,2), (1,3), 
                     (2,4), (3,5), (4,6), (1,7), 
                     (2,8), (7,8), (7,9), (8,10), 
                     (9,11), (10,12)]

skeletons['lsp'] = [(12, 13), (12, 8), (12, 9), (8, 7),
                     (7, 6), (9, 10), (10, 11), (8, 2),
                     (9, 3), (2, 1), (1, 0), (3, 4),
                     (4, 5), (2,3), (9,8)]

skeletons['mpii'] = [(0,1), (1,2), (2,6), (6,3), 
                     (3,4), (4,5), (6,7), (7,8), 
                     (8,9), (13, 7), (7, 12), (10,11), 
                     (11,12), (15,14), (14,13)]

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--tgt", default='lsp')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
    config['model_params']['KTS_params']['edges'] = opt.tgt
    if opt.tgt == 'mpii':
        config['model_params']['KTS_params']['n_kps'] = 16
        config['model_params']['STK_params']['n_kp'] = 16
    if opt.tgt == 'lsp':
        config['model_params']['KTS_params']['n_kps'] = 14
        config['model_params']['STK_params']['n_kp'] = 14
    if opt.tgt == 'h36m':
        config['model_params']['KTS_params']['n_kps'] = 15
        config['model_params']['STK_params']['n_kp'] = 15

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    logger = Logger(log_dir, save_frequency=50)

    ##### Model instantiation
    obj_edges = opt.tgt 
    config_kp_to_skl = config['model_params']['KTS_params']
    config_kp_to_skl['edges'] = skeletons[obj_edges]
    config_kp_to_skl['device'] = 'cuda'
    model_kp_to_skl = KPToSkl(**config_kp_to_skl)
    model_kp_to_skl.to(opt.device_ids[0])

    model_skeleton_to_kp = SklToKP(**config['model_params']['STK_params']) 
    model_skeleton_to_kp.to(opt.device_ids[0]) 

    ##### Dataset loading
    
    config['train_params']['dataset'] = opt.tgt
    if opt.tgt == 'penn':
        loader_tgt = LoadPennAction(**config['datasets']['penn_train']) 
        config['datasets']['skeletons']['kp_map'] = HUMANS_TO_PENN
    elif opt.tgt == 'mpii':
        loader_tgt = LoadMpii(**config['datasets']['mpii_train']) 
        config['datasets']['skeletons']['kp_map'] = HUMANS_TO_MPII
    elif opt.tgt == 'lsp':
        loader_tgt = LoadLsp(**config['datasets']['lsp_train']) 
        config['datasets']['skeletons']['kp_map'] = HUMANS_TO_LSP
    loader_src = LoadSkeletonDataset(**config['datasets']['skeletons'])
    tgt_batch = next(iter(loader_tgt))
    debug_encoder(model_skeleton_to_kp,
                       model_kp_to_skl,
                       loader_src,
                       loader_tgt,
                       config['train_params'],
                       opt.checkpoint,
                       logger, opt.device_ids, tgt_batch)
