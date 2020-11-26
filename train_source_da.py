import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy

from modules.networks import KPDetector, MultiScaleDiscriminator
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset
from datasets.penn_action import LoadPennAction
from datasets.lsp import LoadLsp
from datasets.mpii_loader import LoadMpii
from datasets.unaligned_loader import LoadUnalignedH36m 
from datasets.utils import DatasetLoaders, MapH36mTo
from src_tgt_da import train_kpdetector

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--log_dir", default="", help="path to log dir")
    parser.add_argument("--device_ids", default="0", 
                         type=lambda x: list(map(int, x.split(','))), 
                         help="Names of the devices comma separated.")
    parser.add_argument("--src_dataset", default="h36m", type=str, help="which dataset to use for training. [h36m | penn]")
    parser.add_argument("--tgt_dataset", default="mpii", type=str, help="which dataset to use for target")
    parser.add_argument("--test", action="store_true", help='test instead of train model')
    parser.add_argument("--epochs", default=50, type=int, help="nubmer of epochs to train")
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
    config['train_params']['num_epochs'] = opt.epochs

    if opt.src_dataset == "penn":
        config['model_params']['kp_detector_params']['num_kp'] = 13
    elif opt.src_dataset == "lsp":
        config['model_params']['kp_detector_params']['num_kp'] = 14
    elif opt.src_dataset == "mpii":
        config['model_params']['kp_detector_params']['num_kp'] = 16
    elif opt.src_dataset == "h36m":
        config['model_params']['kp_detector_params']['num_kp'] = 32

    train_params = config['train_params']
    kp_map = None
    model_kp_detector = KPDetector(**config['model_params']['kp_detector_params']) 
    model_kp_detector.to(opt.device_ids[0]) 

    model_discriminator= None
    if train_params['use_gan']:
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

    train_kpdetector(model_kp_detector,
                       loaders,
                       train_params,
                       opt.checkpoint,
                       logger, opt.device_ids, 
                       model_discriminator=model_discriminator,
                       kp_map=kp_map)
 
