import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from argparse import ArgumentParser
import yaml
from time import gmtime, strftime
from shutil import copy

from modules.networks import ConditionalImageGenerator, KPToSkl
from tensor_logger import Logger
from datasets.humans36m import LoadHumansDataset
from datasets.unaligned_loader import LoadUnalignedH36m 
from debug.conditional_generator import debug_generator

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

    logger = Logger(log_dir)

    ##### Model instantiation
    obj_edges = config['model_params']['KTS_params']['edges']
    config_kp_to_skl = config['model_params']['KTS_params']
    config_kp_to_skl['edges'] = skeletons[obj_edges]
    config_kp_to_skl['device'] = 'cuda'
    model_kp_to_skl = KPToSkl(**config_kp_to_skl)
    model_kp_to_skl.to(opt.device_ids[0])

    model_generator = ConditionalImageGenerator(**config['model_params']['generator_params'])
    model_generator.to(opt.device_ids[0])

    loader = LoadHumansDataset(**config['datasets']['h36m_source'])
    loader_tgt = LoadHumansDataset(**config['datasets']['h36m_target'])
    tgt_batch = next(iter(loader_tgt))
    ##### Train
    debug_generator(model_generator,  
           model_kp_to_skl,
           loader,
           config['train_params'],
           logger,
           opt.device_ids,
           tgt_batch=tgt_batch)

