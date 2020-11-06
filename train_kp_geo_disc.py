import os
import sys
import yaml
from time import gmtime, strftime
from argparse import ArgumentParser
from shutil import copy
import torch

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from modules.networks import KPToSkl, Discriminator2D
from modules.networks import MultiScaleDiscriminator 
from modules.networks import KPDetectorVerbose, KPDetector

from datasets.humans36m import LoadHumansDataset, batch_fn
from datasets.penn_action import LoadPennAction, batch_penn
from datasets.mpii_loader import LoadMpii
from datasets.couple_loader import LoadCoupledDatasets
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.lsp import LoadLsp
from datasets.annot_converter import HUMANS_TO_LSP, HUMANS_TO_MPII 
from datasets.annot_converter import HUMANS_TO_PENN, MPII_TO_HUMANS

from kp_disc_geo import train_generator_geo

from tensor_logger import Logger


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--src_model", required=True, help="Model to adapt")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument('--epochs', default=100, help="number of epochs")
    parser.add_argument("--tgt", default='lsp')
    parser.add_argument("--geo", default=1, help="geo loss")
    parser.add_argument("--gamma", default=800, help="gamma")
    parser.add_argument("--save_if", default=0.4, type=float, help="save_if")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
    config['train_params']['loss_weights']['geometric'] = float(opt.geo)
    config['train_params']['num_epochs'] = int(opt.epochs)
    config['model_params']['KTS_params']['edges'] = opt.tgt
    config['model_params']['KTS_params']['gamma'] = float(opt.gamma)
    config['train_params']['save_if'] = float(opt.save_if)
    if opt.tgt == 'penn':
        config['model_params']['KTS_params']['n_kps'] = 13
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 13
    if opt.tgt == 'mpii':
        config['model_params']['KTS_params']['n_kps'] = 16
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 16
    if opt.tgt == 'lsp':
        config['model_params']['KTS_params']['n_kps'] = 14
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 14
    if opt.tgt == 'humans':
        config['model_params']['KTS_params']['n_kps'] = 15
        config['model_params']['discriminator_heatmap_params']['num_channels'] = 15

    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
    log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime()) 

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    logger = Logger(log_dir, save_frequency=5)

    ##### Model instantiation
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 

    if opt.src_model != 'scratch':
        try:
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
    #model_discriminator = MultiScaleDiscriminator(config['model_params']['discriminator_heatmap_params'], scales=[1, 0.5, 0.25])
    model_discriminator = MultiScaleDiscriminator(config['model_params']['discriminator_heatmap_params'], scales=[1.])
    model_discriminator.to(opt.device_ids[0])

    disc_params = config['model_params']['discriminator_params']
    obj_edges = opt.tgt


    ##### Dataset loading
    loader_src = LoadHumansDataset(**config['datasets']['h36m_resized_crop_protocol2_train'])
    config['train_params']['dataset'] = opt.tgt
    if opt.tgt == 'penn':
        loader_tgt = LoadPennAction(**config['datasets']['penn_train']) 
        loader_test = LoadPennAction(**config['datasets']['penn_test'])
        kp_map = HUMANS_TO_PENN
    elif opt.tgt == 'mpii':
        loader_tgt = LoadMpii(**config['datasets']['mpii_train']) 
        loader_test = LoadMpii(**config['datasets']['mpii_eval'])
        kp_map = HUMANS_TO_MPII
    elif opt.tgt == 'lsp':
        loader_tgt = LoadLsp(**config['datasets']['lsp_train']) 
        loader_test = LoadLsp(**config['datasets']['lsp_test'])
        kp_map = HUMANS_TO_LSP
    elif opt.tgt == 'humans':
        loader_tgt = LoadHumansDataset(**config['datasets']['h36m_resized_simplified_train']) 
        loader_test = LoadHumansDataset(**config['datasets']['h36m_resized_simplified_test'])
        kp_map = MPII_TO_HUMANS 
        #kp_map = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 19, 25, 26, 27]
 

    train_generator_geo(model_kp_detector,
                         model_discriminator,
                         loader_src,
                         loader_tgt,
                         loader_test,
                         config['train_params'],
                         opt.checkpoint,
                         logger, opt.device_ids, kp_map)
