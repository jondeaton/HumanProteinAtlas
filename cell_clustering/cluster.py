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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture

from HumanProteinAtlas import Dataset, Sample
from partitions import Split, partitions
from preprocessing import preprocess_dataset
from deep_model.config import Configuration
from feature_extraction import batch_fit_pca_model, batch_disk_pca_transform
from evaluation.metrics import plot_histogram

from cell_clustering.predict import predict_sample_prob


class ClusteringMethod(Enum):
    kmeans = 0
    gmm = 1


def train_gmm(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(X)

    return gmm


def train_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans


def assign_samples(ids, model, X):
    assert isinstance(model, GaussianMixture)

    predictions = model.predict_proba(X)
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

    n_clusters = 27

    # Extract / Fit

    pca_model = batch_fit_pca_model(human_protein_atlas, partitions.train, save_dir=args.features_dir)

    # Transform

    pca_features = batch_disk_pca_transform(pca_model, args.features_dir)

    # Clear some memory
    pca_model = None

    # Train GMM model

    logger.info("Fitting cell clusters...")
    model = None
    if os.path.exists(args.model_file):
        logger.info("Loading model from file.")
        with open(args.model_file, "rb") as file:
            model = pickle.load(file)
    else :
        model = train_gmm(pca_features, n_clusters)
        logger.info("Saving model.")
        pickle.dump(model, open(args.model_file, 'wb+'))

    # Predict cluster probabilities

    logger.info("Assigning training set clusters...")
    assignments = None
    if os.path.exists(args.assignments_file):
        logger.info("Loading model from file.")
        with open(args.assignments_file, "rb") as file:
            assignments = pickle.load(file)
    else :
        assignments = assign_samples(partitions.train, model, pca_features)
        logger.info("Saving cluster assignments")
        pickle.dump(assignments, open(args.assignments_file, 'wb+'))

    # TODO: update to fit with new changes

    # Clustering check

    # Get PCA top two features
    # X = None
    # X = extract_features(human_protein_atlas, partitions.train, 2, args.features_dir)

    # predictions = train_gmm(X, n_clusters).predict(X)

    # # Plot clusters in 2D

    # color_range = cm.rainbow(np.linspace(0, 1, n_clusters))
    # colors = [color_range[predictions[i]] for i in range(len(predictions))]
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.show()

    # # Compare counts in clusters to Kmeans
    # predictions_kmeans = train_kmeans(X, n_clusters).predict(X)

    # counts = np.zeros((2, n_cells))
    # for i in range(len(predictions)):
    #     counts[0, predictions[i]] += 1
    #     counts[1, predictions_kmeans[i]] += 1
    
    # plot_histogram(counts, np.arange(n_cells), ["GMM", "KMeans"], "Cell clusters", "Cluster index", \
    #     "Number of samples", "outputs/cluster_counts.png")

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
    output_options.add_argument("--features-dir", required=True, help="Directory to save pca/radon extracted features in")
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
