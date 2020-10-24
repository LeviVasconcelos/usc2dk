import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy

from modules.networks import KPDetector
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset
from datasets.penn_action import LoadPennAction
from datasets.lsp import LoadLsp
from datasets.mpii_loader import LoadMpii
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.annot_converter import HUMANS_TO_MPII, HUMANS_TO_LSP, HUMANS_TO_PENN
from source_kp_detector import train_kpdetector
from nips.MTFAN import FAN

skeletons = {}
skeletons['humans'] = [(0,1), (1,2), (2,3), (0, 4), 
                        (4,5), (5,6), (0,7), (7,8), 
                        (7,9), (9,10), (10, 11), (7,12), 
                        (12, 13), (13, 14)] 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--dataset", default="h36m", type=str, help="which dataset to use for training. [h36m | penn]")
    parser.add_argument("--tgt_model", default=None, type=str, help="for nips comparisson: name of dataset to build the model for (although training on --dataset")
    parser.add_argument("--test", action="store_true", help='test instead of train model')
    parser.add_argument("--fan", action="store_true", help='model is fan')
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

    logger = Logger(log_dir, save_frequency=50)

    if opt.dataset == "penn":
        config['model_params']['kp_detector_params']['num_kp'] = 13
    elif opt.dataset == "lsp":
        config['model_params']['kp_detector_params']['num_kp'] = 14
    elif opt.dataset == "mpii":
        config['model_params']['kp_detector_params']['num_kp'] = 16
    elif opt.dataset == "h36m":
        config['model_params']['kp_detector_params']['num_kp'] = 32

    kp_map = None
    if opt.tgt_model == "mpii":
        config['model_params']['kp_detector_params']['num_kp'] = 16
        config['datasets']['h36m_resized_crop_protocol2_train']['kp_list'] = HUMANS_TO_MPII
        config['datasets']['h36m_resized_crop_protocol2_target']['kp_list'] = HUMANS_TO_MPII
    ##### Model instantiation
    if opt.fan:
        model_kp_detector = FAN(1, n_points=config['model_params']['kp_detector_params']['num_kp'])
        print('opt fan: heatmapres: ', model_kp_detector.heatmap_res)
    else:
        model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 

    ##### Dataset loading
    dataset_source = ""
    dataset_eval = ""
    if opt.dataset == "h36m":
        dataset_source = 'h36m_resized_crop_protocol2_train'
        dataset_eval = 'h36m_resized_crop_protocol2_target'
        loader = LoadHumansDataset(**config['datasets'][dataset_source])
        loader_tgt = LoadHumansDataset(**config['datasets'][dataset_eval])
    elif opt.dataset == "penn":
        dataset_source = 'penn_train'
        dataset_eval = 'penn_test'
        loader = LoadPennAction(**config['datasets'][dataset_source])
        loader_tgt = LoadPennAction(**config['datasets'][dataset_eval])
    elif opt.dataset == "lsp":
        dataset_source = 'lsp_train'
        dataset_eval = 'lsp_test'
        loader = LoadLsp(**config['datasets'][dataset_source])
        loader_tgt = LoadLsp(**config['datasets'][dataset_eval])
    elif opt.dataset == 'mpii':
        dataset_source = 'mpii_train'
        dataset_eval = 'mpii_eval'
        loader = LoadMpii(**config['datasets'][dataset_source])
        loader_tgt = LoadMpii(**config['datasets'][dataset_eval]) 
    tgt_batch = next(iter(loader_tgt))
    config['train_params']['dataset'] = opt.dataset
    config['train_params']['test'] = opt.test
    train_kpdetector(model_kp_detector,
                       loader,
                       loader_tgt,
                       config['train_params'],
                       opt.checkpoint,
                       logger, opt.device_ids, tgt_batch)
