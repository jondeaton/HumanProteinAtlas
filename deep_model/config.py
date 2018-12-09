#!/usr/bin/env python
"""
File: config.py
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""
import os
import configparser

dir_name = os.path.dirname(__file__)
default_config_file = os.path.join(dir_name, "config.ini")


class Configuration(object):

    def __init__(self, config_file=default_config_file):
        assert isinstance(config_file, str)
        # Setup the filesystem configuration
        self._config_file = os.path.join(config_file)
        self._config = configparser.ConfigParser()
        self._config.read(self._config_file)
        c = self._config

        self.gmm_model_file = os.path.expanduser(c["GMM"]["path"])

        self.dataset_directory = os.path.expanduser(c["Data"]["path"])
        self.model_file = os.path.expanduser(c["Output"]["save-file"])
        self.save_freq = int(c['Output']['save_freq'])

        self.tensorboard_dir = os.path.expanduser(c["TensorFlow"]["tensorboard-dir"])
        self.tensorboard_freq = int(c["TensorFlow"]["log-frequency"])

    def override(self, args):
        for attr in vars(args):
            if getattr(self, attr) is not None:
                value = getattr(args, attr)
                setattr(self, attr, value)