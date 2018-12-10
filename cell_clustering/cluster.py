#!/usr/bin/env python
"""
File: cluster
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os, sys
import argparse
import logging
import pickle

import numpy as np
from sklearn.mixture import GaussianMixture

import HumanProteinAtlas
from HumanProteinAtlas import Dataset, Sample, Color

from partitions import Split, partitions
from deep_model.config import Configuration
from feature_extraction import Feature, get_features

import multiprocessing as mp


def train_gmm(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(X)
    return gmm


def _get_ft(t):
    return get_ft(*t)


def get_ft(dataset, id):
    print("Extracting features for: %s" % id)
    img = dataset[id].combined((Color.blue, Color.yellow, Color.red))
    return get_features(img, method=Feature.dct)


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default

    np.random.seed(args.seed)

    if args.recompute:
        logger.debug("Dataset location: %s" % config.dataset_directory)
        human_protein_atlas = HumanProteinAtlas.Dataset(config.dataset_directory)

        sampled_ids = np.random.choice(partitions.train, args.num_examples, replace=False)

        logger.info("Extracting features from %d examples" % args.num_examples)
        arguments = [(human_protein_atlas, id) for id in sampled_ids]
        pool = mp.Pool(4)
        features = pool.map(_get_ft, arguments)

        X = np.vstack(features)
        if args.features_file is not None:
            logger.info("Saving features to: %s" % args.features_file)
            np.save(args.features_file, X)
    else:
        logger.info("Loading features from: %s" % args.features)
        X = np.load(args.features_file)

    logger.info("Fitting GMM Model...")
    model = train_gmm(X, args.num_clusters)
    logger.info("GMM model fit.")

    logger.info("Saving GMM model to: %s" % args.model_file)
    with open(args.model_file, 'wb+') as f:
        pickle.dump(model, f)


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
