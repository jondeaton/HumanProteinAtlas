#!/usr/bin/env python
"""
File: __init__.py
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
from partitions.partition import Split

default_partition_store = os.path.join(os.path.split(__file__)[0],
                                       "cluster_ouputs")

default_locations = {

    ClusteringMethod.kmeans: {Split.train: os.path.join(default_partition_store, "kmeans_train"),
                              Split.test: os.path.join(default_partition_store, "kmeans_test")},

    ClusteringMethod.gmm: {Split.train: os.path.join(default_partition_store, "gmm_train"),
                           Split.test: os.path.join(default_partition_store, "gmm_test")},
}
