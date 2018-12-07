#!/usr/bin/env python
"""
File: cluster
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
import sys
import argparse
import logging
import datetime
import pickle
from enum import Enum

import numpy as np
from HumanProteinAtlas import Dataset, Sample
from partitions import Split, partitions
from preprocessing import preprocess_dataset
from deep_model.config import Configuration
from feature_extraction import compute_radon_features

def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default
    
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)
    human_protein_atlas = Dataset(config.dataset_directory)

    make_radon_dataset(human_protein_atlas, partitions.train, args.features_dir, args.out_dir)

    logger.info("Exiting.")


def make_radon_dataset(human_protein_atlas, ids, in_dir, out_dir):
    outfile_names = compute_radon_features(human_protein_atlas, ids, in_dir)

    ids = np.array_split(ids, 10)

    for i, name in enumerate(outfile_names):
        print("Handling file:", name)
        radon_data_i = None
        with open(name, "rb" ) as file:
            radon_data_i = pickle.load(file)

        for j, id in enumerate(ids[i]):
            id_save_file = os.path.join(out_dir, str(id) + ".radon_data")
            with open(id_save_file, 'wb+') as file:
                np.save(file, radon_data_i[j])


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster training data into cell types",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Config")
    info_options.add_argument("--config", type=str, help="Configuration file")
    info_options.add_argument("-d", "--num-clusters", default=4, type=int, help="Number of clusters")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--path', help="Dataset input file")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--features-dir", required=True, help="Directory to save pca/radon extracted features in")
    output_options.add_argument("--out-dir", required=True, help="Directory to save features in")

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
