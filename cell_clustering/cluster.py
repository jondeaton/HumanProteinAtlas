#!/usr/bin/env python
"""
File: models
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
import sys
import argparse
import logging
import datetime
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture

from HumanProteinAtlas import Dataset, Sample
from partitions import Split, partitions
from preprocessing import preprocess_dataset
from deep_model.config import Configuration

import cell_clustering
from feature_extraction import extract_features

import pickle


class ClusteringMethod(Enum):
    kmeans = 0
    meanshift = 1
    gmm = 2


def train_gmm(X, n_cluster):
    gmm = GaussianMixture(n_components=n_cluster)
    gmm.fit(X)
    return gmm


def assign_samples(human_protein_atlas, ids, model):
    assert isinstance(human_protein_atlas, Dataset)
    assert isinstance(model, GaussianMixture)

    X = extract_features(human_protein_atlas, ids)
    predictions = model.predict(X)
    assignments = dict()
    for i, id in enumerate(ids):
        assignments[id] = predictions[i]
    return assignments


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default
    
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)
    human_protein_atlas = Dataset(config.dataset_directory)

    X = extract_features(human_protein_atlas, partitions.train)

    logger.info("Fitting cell clusters...")
    model = train_gmm(X, partitions.train)

    logger.info("Saving GMM model.")
    pickle.dump(model, open(args.model_file, 'wb'))

    logger.info("Assigning training set clusters...")
    assignments = assign_samples(human_protein_atlas, partitions.train, model)

    logger.info("Saving cluster assignments")
    pickle.dump(assignments, open(args.assignments_file, 'wb'))

    logger.info("Exiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster training data into cell types",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Config")
    info_options.add_argument("--config", type=str, help="Configuration file")
    info_options.add_argument("-d", "--num-clusters", default=4, type=int, help="Number of clusters")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--path', help="Dataset input file")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", required=True, help="File to save trained model in")
    output_options.add_argument("--assignments-file", required=True, help="Save assignments")

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
