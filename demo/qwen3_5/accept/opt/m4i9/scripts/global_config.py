"""Global Configuration"""

import os


class GlobalConfig:
    def __init__(self):
        # Verbose
        self.verbose = False

        # default gpu device id
        self.gpu_device_id = 0

        # bypass compile errors
        self.bypass_compile_errors = False


global_config = GlobalConfig()
