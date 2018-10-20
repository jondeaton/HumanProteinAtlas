#!/usr/bin/env python
"""
File: __init__.py
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
from partitions.partition import Split, Partitions

# some default locations
default_partition_store = os.path.join(os.path.split(__file__)[0],
                                       "HumanProteinAtlas-partitions")

default_locations = {
    Split.train: os.path.join(default_partition_store, "train"),
    Split.test: os.path.join(default_partition_store, "test"),
    Split.validation: os.path.join(default_partition_store, "validation")
}


default_splits = Partitions(default_locations)
