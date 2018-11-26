#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from HumanProteinAtlas import Dataset


def extract_features(human_protein_atlas, ids):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    m = len(ids)
    d = 10  # for instance
    X = np.empty((m, d))

    for i, id in enumerate(ids):
        sample = human_protein_atlas[id]
        features = image_extract_features(sample.multi_channel)
        X[i, :] = features

    return X


def image_extract_features(image):
    assert isinstance(image, np.ndarray)
    # todo: actually extract some useful features
    return image.flatten()

