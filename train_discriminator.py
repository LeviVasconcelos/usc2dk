import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy
import torch

import numpy as np
from modules.networks import KPDetector, MultiScaleDiscriminator
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset
from datasets.penn_action import LoadPennAction
from datasets.lsp import LoadLsp
from datasets.mpii_loader import LoadMpii
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.utils import DatasetLoaders, MapH36mTo
from disciminator_train import train_discriminator

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--src_dataset", default="mpii", type=str, help="which dataset to use for training. [h36m | penn]")
    parser.add_argument("--tgt_dataset", default="penn", type=str, help="which dataset to use for target")
    parser.add_argument("--src_model", default=None,  help="pretrained model on source")
    parser.add_argument("--epochs", default=500, type=int, help="nubmer of epochs to train")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    logger = Logger(log_dir, save_frequency=25)

    config['train_params']['num_epochs'] = opt.epochs
    if opt.src_dataset == "penn":
        config['model_params']['kp_detector_params']['num_kp'] = 13
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 13
    elif opt.src_dataset == "mpii":
        config['model_params']['kp_detector_params']['num_kp'] = 16
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 16

    train_params = config['train_params']
    kp_map = None
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 
    
    if opt.src_model is not None:
        try:
            adapt_pretrained = True
            print(f"loading {opt.src_model}")
            kp_state_dict = torch.load(opt.src_model)
            print('Source model loaded: %s' % opt.src_model)
        except:
            print('Failed to read model %s' % opt.src_model)
            exit(1)
        try:
            model_kp_detector.load_state_dict(kp_state_dict['model_kp_detector'])
        except:
            print('failed to load model weights')
            exit(1)

    #label_generator.convert_bn_to_dial(label_generator, device= opt.device_ids[0])
    #label_generator = label_generator.cuda()#label_generator.to(opt.device_ids[0]) 
    
    model_discriminator = MultiScaleDiscriminator(config['model_params']['discriminator_heatmap_params'], scales=[1.])
    model_discriminator.to(opt.device_ids[0])
    
    ##### Dataset loading
    src_dset_loader = DatasetLoaders[opt.src_dataset]
    tgt_dset_loader = DatasetLoaders[opt.tgt_dataset]
    cfg_dset = config['datasets']
    loader_src_train = src_dset_loader(**cfg_dset[train_params['src_train']])
    loader_src_test = src_dset_loader(**cfg_dset[train_params['src_test']])
    loader_tgt_train = tgt_dset_loader(**cfg_dset[train_params['tgt_train']])
    #loader_tgt_test = tgt_dset_loader(**cfg_dset[train_params['tgt_test']])
    loaders = [loader_src_train, loader_src_test, loader_tgt_train]

    kp_map = MapH36mTo[opt.tgt_dataset]
    MPII_TO_PENN=np.array([9,13,12,14,11,15,10,3,2,4,1,5,0])
    kp_map = MPII_TO_PENN

    train_discriminator(model_kp_detector,
                       model_discriminator,
                       loaders,
                       train_params,
                       opt.checkpoint,
                       logger, opt.device_ids, 
                       kp_map=kp_map)
 