#!/usr/bin/env python
"""
File: reduce_features
Date: 12/7/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import sys
import numpy as np
import logging, argparse

import HumanProteinAtlas
from deep_model.config import Configuration
from HumanProteinAtlas import Dataset

from partitions import partitions
import multiprocessing as mp



def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()

    output_dir = os.path.expanduser(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dataset = HumanProteinAtlas.Dataset(config.dataset_directory)



def parse_args():
    parser = argparse.ArgumentParser(description="Extract Features",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", required=True, help="Output directory to store plots")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--config", required=False, type=str, help="Configuration file")
    options_group.add_argument("-params", "--params", type=str, help="Hyperparameters json file")
    options_group.add_argument("--scale", action='store_true', help="Scale the images down")
    options_group.add_argument("-p", "--pool-size", type=int, default=8, help="Worker pool size")

    logging_options = parser.add_argument_group("Logging")
    logging_options.add_argument('--log', dest="log_level", default="DEBUG", help="Logging level")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()
