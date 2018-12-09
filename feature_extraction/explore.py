#!/usr/bin/env python
"""
File: explore
Date: 12/7/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import logging, argparse

import numpy as np

import HumanProteinAtlas
from deep_model.config import Configuration
from HumanProteinAtlas import Dataset, Color

from partitions import partitions
from feature_extraction import get_features

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from feature_extraction import Feature
import multiprocessing as mp


"""
Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
"""
def silhouette_plots(X):
    range_n_clusters = [2, 4, 8, 16, 27, 30]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


def explore_features(X):
    # tsne_components = TSNE(n_components=3).fit_transform(X)

    # plt.figure()
    # plt.scatter(tsne_components[:, 0], tsne_components[:, 1])
    # plt.show()

    print("Raw features, dim=", X[0].shape)
    silhouette_plots(X)

    pca = PCA(n_components=2, whiten=True)
    principalComponents = pca.fit_transform(X)

    print("PCA=2:")
    silhouette_plots(principalComponents)

    plt.figure()
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.show()


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
        config = Configuration()

    if args.output is not None:
        output_dir = os.path.expanduser(args.output)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    if not os.path.exists("X.npy"):
        dataset = HumanProteinAtlas.Dataset(config.dataset_directory, scale=True)
        experimental_ids = np.random.choice(partitions.train, args.num_examples, replace=False)

        arguments = [(dataset, id) for id in experimental_ids]
        pool = mp.Pool(8)
        features = pool.map(_get_ft, arguments)

        X = np.vstack(features)

        np.save("X", X)
    else:
        X = np.load("X.npy")

    explore_features(X)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Features",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", required=False, help="Output directory to store plots")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--config", required=False, type=str, help="Configuration file")
    options_group.add_argument("-p", "--pool-size", type=int, default=8, help="Worker pool size")
    options_group.add_argument("-m", "--num-examples", type=int, default=2000, help="Number of examples")

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
