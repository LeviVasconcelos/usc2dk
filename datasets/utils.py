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
from datasets.annot_converter import HUMANS_TO_MPII, HUMANS_TO_LSP, HUMANS_TO_PENN, HUMANS_TO_HUMANS

DatasetLoaders = {
        'lsp': LoadLsp,
        'penn': LoadPennAction,
        'h36m': LoadHumansDataset,
        'mpii': LoadMpii,
       }
MapH36mTo = {
        'lsp': HUMANS_TO_LSP,
        'mpii': HUMANS_TO_MPII,
        'penn': HUMANS_TO_PENN,
        'h36m': HUMANS_TO_HUMANS,
        }


