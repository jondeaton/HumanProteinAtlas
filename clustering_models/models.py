#!/usr/bin/env python
"""
File: models
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture

def main():
    # TODO: testing, adapt to needs
    data = np.array([[1, 2], [3, 4], [5, 6]])
    new_data = np.array([[0, 1], [6, 7]])
    num_cell_types = 3

    # K-means
    kmeans = KMeans(n_clusters=num_cell_types).fit(data)
    cluster_assignments = kmeans.labels_
    predictions = kmeans.predict(new_data)
    centroids = kmeans.cluster_centers_

    # Meanshift
    meanshift = MeanShift(bandwidth=2).fit(data)
    cluster_assignments = meanshift.labels_
    predictions = meanshift.predict(new_data)

    # GMM EM
    gmm = GaussianMixture(n_components=num_cell_types).fit(data)
    cluster_assignments = gmm.predict(new_data)

if __name__ == "__main__":
    main()
