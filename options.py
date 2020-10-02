import argparse
import os

class Options():
    def __init__(self):
        self.myparser = argparser.ArgumentParser()
        self.add_options()

    def add_options(self):
       self.myparser.add_argument("--config", required=True, help="path to config file")
       self.myparser.add_argument("--checkpoint", default=None, help="path to checkpoint")
       self.myparser.add_argument("--log_dir", default="", help="path to log dir")
       self.myparser.add_argument("--device_ids", default="0", 
                                     type=lambda x: list(map(int, x.split(','))), 
                                     help="Names of the devices comma separated.")
       self.myparser.add_argument("--tgt", default='penn',help="Target dataset.")
       self.myparser.add_argument("--epochs", default=5,help="Weight of the source supervised loss.")
       self.myparser.add_argument("--src_model", required=True, help="Model to adapt")
       self.myparser.add_argument("--no_map", action="store_true", help="whether not to use kp mapping") 
       self.myparser.add_argument("--src", default="humans", help="source model")
  
