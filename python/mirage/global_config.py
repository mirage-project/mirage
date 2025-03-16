""" Global Configuration """

import os

class GlobalConfig:
    def __init__(self):
        # Verbose
        self.verbose = False

        # default gpu device id
        self.gpu_device_id = 0

global_config = GlobalConfig()
