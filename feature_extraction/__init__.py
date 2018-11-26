#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from skimage.transform import radon, rescale
from sklearn.decomposition import PCA

from HumanProteinAtlas import Dataset, Sample


def extract_features(human_protein_atlas, ids, d=10, use_radon=True):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    features = None
    if use_radon:
        features = compute_radon_features(human_protein_atlas, ids)
    # else:
    #     for i, id in enumerate(ids):
    #         sample = human_protein_atlas[id]
    #         assert isinstance(sample, Sample)
    #         sample_features = np.concatenate((sample.yellow, sample.red, sample.blue))
    #         extracted_features.append(sample_features)

    #         # TODO: remove! for debugging only
    #         if (i % 10 == 0): print(i)
    #         if i == 449: break

    pca = PCA(n_components=d, whiten=True)
    components = pca.fit_transform(features)
    
    # TODO: change! for testing only
    # assert components.shape == (len(ids), d)
    assert components.shape == (450, d)
 
    return components


def compute_radon_features(human_protein_atlas, ids):
    extracted_features = []    

    for i, id in enumerate(ids):
        sample = human_protein_atlas[id]
        assert isinstance(sample, Sample)

        yellow_features = get_radon_transform(sample.yellow)
        red_features = get_radon_transform(sample.red)
        blue_features = get_radon_transform(sample.blue)
        sample_features = np.concatenate((yellow_features, red_features, blue_features))

        extracted_features.append(sample_features)

        # TODO: remove! for debugging only
        if (i % 10 == 0): print("Sample extract iteration:", i)
        if i == 449: break

    return np.array(extracted_features)


def get_radon_transform(img, n_beams=64):
    # Scale image for desired beam count
    scale = n_beams / np.ceil(np.sqrt(2) * max(img.shape))
    img = rescale(img, scale=scale, mode="reflect")

    sinogram = radon(img, theta=np.arange(360), circle=True)

    return sinogram.flatten()


def image_extract_features(image):
    assert isinstance(image, np.ndarray)
    # todo: actually extract some useful features
    return image.flatten()
