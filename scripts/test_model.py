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
from debug.kp_detector import train_kpdetector
from nips.MTFAN import FAN

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--dataset", default="h36m", type=str, help="which dataset to use for training. [h36m | penn]")
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

    ##### Model instantiation
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    #model_kp_detector = FAN(1, n_points=32)
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
 
