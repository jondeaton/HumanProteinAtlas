#!/usr/bin/env python
"""
File: precompute
Date: 12/9/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import sys
import argparse
import logging
import pickle

import numpy as np

from HumanProteinAtlas import Dataset, Sample, Color

from partitions import Split, partitions
from deep_model.config import Configuration
from feature_extraction import Feature, get_features

import multiprocessing as mp



def _get_gmm_probas(t):
    return get_gmm_probas(*t)

def get_gmm_probas(dataset, id,  gmm_model):
    print("Extracting features for: %s" % id)
    img = dataset[id].combined((Color.blue, Color.yellow, Color.red))
    features = get_features(img, method=Feature.dct)
    features = np.expand_dims(features, axis=0)
    return gmm_model.predict_proba(features)[0]

def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default

    np.random.seed(args.seed)

    logger.info("Loading GMM model from: %s" % config.gmm_model_file)
    with open(config.gmm_model_file, 'rb') as f:
        gmm_model = pickle.load(f)

    human_protein_atlas = Dataset(config.dataset_directory)

    pool = mp.Pool(8)
    arguments = [(human_protein_atlas, id, gmm_model) for id in partitions.train]
    features = pool.map(arguments, _get_gmm_probas)

    feature_map = {partitions.train[i]: features[i] for i in range(len(partitions.train))}
    with open ("train_probas", 'wb+') as f:
        pickle.dump(feature_map, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster training data into cell types",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    hp_args = parser.add_argument_group("HyperParameters")
    hp_args.add_argument("-d", "--num-clusters", default=4, type=int, help="Number of clusters")
    hp_args.add_argument("-m", "--num-examples", default=5000, type=int, help="Number of examples to use")
    hp_args.add_argument("-s", "--seed", type=int, default=0, help="Random seed")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", required=True, help="File to save trained model in")
    output_options.add_argument("--features-file", required=False, help="File to save extracted features in")
    output_options.add_argument("--assignments-file", required=False, help="Save assignments")

    config_args = parser.add_argument_group("Config")
    config_args.add_argument("--recompute", action='store_true', help="Recompute features")
    config_args.add_argument("--config", type=str, help="Configuration file")

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
