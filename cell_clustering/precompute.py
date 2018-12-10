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
from sklearn.mixture import GaussianMixture

import multiprocessing as mp

def _get_gmm_probas(t):
    return get_gmm_probas(*t)

def get_gmm_probas(dataset, id,  gmm_model):
    print("Extracting features for: %s" % id)
    img = dataset[id].combined((Color.blue, Color.yellow, Color.red))
    features = get_features(img, method=Feature.dct)
    features = np.expand_dims(features, axis=0)
    probs = gmm_model.predict_proba(features)[0]
    return probs


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

    id_set = partitions.train + partitions.test

    id_set = np.random.choice(id_set, 50, replace=False)

    pool = mp.Pool(8)
    arguments = [(human_protein_atlas, id, gmm_model) for id in id_set]
    features = pool.map(_get_gmm_probas, arguments)

    print("Saving feature map in: %s" % args.output)
    feature_map = {id_set[i]: features[i] for i in range(len(id_set))}
    with open (args.output, 'wb+') as f:
        pickle.dump(feature_map, f)

    X = np.vstack([feature_map[id] for id in partitions.train if id in feature_map])

    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=8)
    gmm.fit(X)

    X_all = np.vstack(features)
    probas = gmm.predict_proba(X_all)

    print("Saving probas map: %s" % "probas_map")
    probas_map = {id_set[i]: probas[i] for i in range(len(id_set))}
    with open("probas_map", "wb+") as f:
        pickle.dump(probas_map, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster training data into cell types",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    hp_args = parser.add_argument_group("HyperParameters")
    hp_args.add_argument("-s", "--seed", type=int, default=0, help="Random seed")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", required=True, help="File to save thing in")

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
