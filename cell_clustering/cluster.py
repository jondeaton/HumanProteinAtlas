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


class ClusteringMethod(Enum):
    kmeans = 0
    meanshift = 1
    gmm = 2


def load_dataset(dataset, split):
    assert isinstance(dataset, Dataset)
    assert isinstance(split, Split)

    sample_data =  []
    sample_ids = []
    for sample_id in partitions[split]:
        sample = dataset.sample(sample_id)
        assert isinstance(sample, Sample)
        sample_data.append(sample.multi_channel.flatten())
        sample_ids.append(sample_id)

    return [np.array(sample_data), sample_ids]


def create_data_pipeline(human_protein_atlas):
    datasets = [load_dataset(human_protein_atlas, split)
                for split in (Split.train, Split.test)]

    n_datasets = len(datasets)

    data = [datasets[i][0] for i in range(n_datasets)]
    ids = [datasets[i][1] for i in range(n_datasets)]

    for i in range(n_datasets):
        data[i] = preprocess_dataset(data[i])

    train_data, test_data = data
    train_ids, test_ids = ids

    return train_data, train_ids, test_data, test_ids


def train(data, ids, method=ClusteringMethod.kmeans, n_clusters=27):
    logger.info("Begin clustering fit.")
    model = None

    if (method == ClusteringMethod.kmeans):
        kmeans = KMeans(n_clusters=n_clusters).fit(data)
        cluster_assignments = kmeans.labels_
        # centroids = kmeans.cluster_centers_

        # TODO: fix errors with this
        # out_file = cell_clustering.default_locations[ClusteringMethod.kmeans][Split.train]
        # with open(out_file, 'w') as f:
        #     for i in range(len(ids)):
        #         f.write(str(ids[i]) + " " + str(cluster_assignments[i]) + "\n")

        model = kmeans
    elif method == ClusteringMethod.meanshift:
        # meanshift = MeanShift(bandwidth=2).fit(data)
        # cluster_assignments = meanshift.labels_
        raise Exception("Not yet implemented.")
    elif method == ClusteringMethod.gmm:
        # gmm = GaussianMixture(n_components=num_cell_types).fit(data)
        raise Exception("Not yet implemented.")
    else:
        raise Exception("Invalid clustering method.")

    logger.info("Exit clustering fit.")

    return model


def predict(data, ids, model, method=ClusteringMethod.kmeans):
    logger.info("Begin clustering prediction.")
    predictions = model.predict(data)

    if (method == ClusteringMethod.kmeans):
        # TODO: fix errors with this
        # out_file = cell_clustering.default_locations[ClusteringMethod.kmeans][Split.test]
        # with open(out_file, 'w') as f:
        #     for i in range(len(ids)):
        #         f.write(str(ids[i]) + " " + str(predictions[i]) + "\n")

        return predictions
    elif method == ClusteringMethod.meanshift:
        raise Exception("Not yet implemented.")
    elif method == ClusteringMethod.gmm:
        raise Exception("Not yet implemented.")
    else:
        raise Exception("Invalid clustering method.")

    logger.info("Exit clustering prediction.")


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default
    
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)
    human_protein_atlas = Dataset(config.dataset_directory)

    train_dataset, train_ids, test_dataset, test_ids  = create_data_pipeline(human_protein_atlas)

    logger.info("Fitting cell clusters...")
    model = train(train_dataset, train_ids)

    logger.info("Predicting test set clusters...")
    predict(test_dataset, test_ids, model)

    logger.info("Exiting.")


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Cluster training data into cell types",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--config", required=False, type=str, help="Configuration file")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--path', help="Dataset input file")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", help="File to save trained model in")

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
