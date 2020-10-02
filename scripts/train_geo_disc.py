import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy
from modules.networks import KPToSkl, Discriminator2D, MultiScaleDiscriminator, ConditionalImageGenerator
from datasets.annot_converter import HUMANS_TO_LSP, HUMANS_TO_MPII, HUMANS_TO_PENN
from modules.networks import KPDetectorVerbose, KPDetector
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset, batch_fn
from datasets.penn_action import LoadPennAction, batch_penn
from datasets.mpii_loader import LoadMpii
from datasets.couple_loader import LoadCoupledDatasets
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.lsp import LoadLsp
from debug.kp_detector_disc_geo import train_generator_geo
from modules.proto_utils import *
from modules.loss_utils import *

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
    parser.add_argument("--src_model", required=True, help="Model to adapt")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument('--epochs', default=50, help="number of epochs")
    parser.add_argument("--tgt", default='lsp')
    parser.add_argument("--geo", default=1, help="geo loss")
    parser.add_argument("--gamma", default=800, help="gamma")
    parser.add_argument("--inpaint", action='store_true', help="do inpaint")
    parser.add_argument("--recover", action='store_true', help="do recovery")
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

    config['train_params']['do_inpaint'] = opt.inpaint
    config['train_params']['do_recover'] = opt.recover
    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
    log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime()) 

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    logger = Logger(log_dir, save_frequency=50)

    ##### Model instantiation
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 
    if opt.src_model != 'scratch':
        try:
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
    
    model_discriminator = MultiScaleDiscriminator(config['model_params']['discriminator_heatmap_params'], scales=[1, 0.5, 0.25])
    model_discriminator.to(opt.device_ids[0])

    disc_params = config['model_params']['discriminator_params']
    #obj_edges = config['model_params']['KTS_params']['edges']
    obj_edges = opt.tgt
    config_kp_to_skl = config['model_params']['KTS_params']
    config_kp_to_skl['edges'] = skeletons[obj_edges]
    config_kp_to_skl['device'] = 'cuda'
    model_kp_to_skl = KPToSkl(**config_kp_to_skl)
    model_kp_to_skl.to(opt.device_ids[0])

    conditional_generator = None
    if opt.inpaint:
        conditional_generator = ConditionalImageGenerator(**config['model_params']['conditional_generator_params'])
        conditional_generator.to(opt.device_ids[0])
    if opt.recover:
        config['model_params']['conditional_generator_params']['hint_features'] = config['model_params']['discriminator_heatmap_params']['num_channels']
        conditional_generator = ConditionalImageGenerator(**config['model_params']['conditional_generator_params'])
        conditional_generator.to(opt.device_ids[0])
 
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
        kp_map = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 19, 25, 26, 27]
 

    train_generator_geo(model_kp_detector,
                         model_discriminator,
                         conditional_generator,
                         model_kp_to_skl,
                         loader_src,
                         loader_tgt,
                         loader_test,
                         config['train_params'],
                         opt.checkpoint,
                         logger, opt.device_ids, kp_map)
