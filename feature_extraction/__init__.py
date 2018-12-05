#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import multiprocessing as mp
import pickle

import numpy as np
from skimage.transform import radon, rescale
from sklearn.decomposition import PCA

from HumanProteinAtlas import Dataset, Sample


def extract_features(human_protein_atlas, ids, d=10, save_dir=".", force_recompute=False):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    pca_file = os.path.join(save_dir, "pca_features_" + str(d))
    if os.path.isfile(pca_file) and not force_recompute:
        return pickle.load(open(pca_file, "rb" ))

    features = None
    if os.listdir(save_dir) and not force_recompute:
        features = pickle.load(open(save_dir, "rb" ))
    else:
        features = compute_radon_features(human_protein_atlas, ids, save_dir)

    pca = PCA(n_components=d, whiten=True)
    components = pca.fit_transform(features)

    # TODO: change! for testing only
    # assert components.shape == (len(ids), d)
    assert components.shape == (100, d)

    pickle.dump(components, open(pca_file, 'wb+'))
 
    return components


def compute_radon_features(human_protein_atlas, ids, save_dir):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    # TODO: remove once done debugging
    ids = ids[:100]

    # Spawn process to handle subset of ids extraction
    processes = []

    # Get CPU count
    try: 
        n_cpus = len(os.sched_getaffinity(0)) # useable cpu count, should work on Unix-base
    except:
        n_cpus = mp.cpu_count() # total cpu count, works on Windows

    for i, id_subset in enumerate(np.array_split(ids, n_cpus)):
        filename = os.path.join(save_dir, "radon_features_" + str(i) + ".radon_data")
        
        p = mp.Process(target=radon_features_helper, args=(human_protein_atlas, id_subset, filename))
        processes.append(p)
        p.start()
    
    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Combine data from save directory and remove files
    # Iterates in order to preserve ids ordering
    combined_features = []
    for i in range(n_cpus):
        filename  = os.path.join(save_dir, "radon_features_" + str(i) + ".radon_data")
        
        assert os.path.isfile(filename)
        assert os.path.getsize(filename) > 0

        with open(filename, "rb" ) as file:
            combined_features = combined_features + pickle.load(file)
        os.remove(filename)

    combined_features = np.array(combined_features)

    print(combined_features.shape)
    print(type(combined_features))

    # Save combined results
    combined_filename = os.path.join(save_dir, "radon_features_combined")
    pickle.dump(combined_features, open(combined_filename, 'wb+'))

    return combined_features


def radon_features_helper(human_protein_atlas, ids, save_file):
    assert isinstance(human_protein_atlas, Dataset)

    extracted_features = []

    for i, id in enumerate(ids):
        sample = human_protein_atlas[id]
        assert isinstance(sample, Sample)

        yellow_features = get_radon_transform(sample.yellow)
        red_features = get_radon_transform(sample.red)
        blue_features = get_radon_transform(sample.blue)
        sample_features = np.concatenate((yellow_features, red_features, blue_features))

        extracted_features.append(sample_features)

        if i % 50 == 0: 
            print("Progress tick, processed images = ", i, "pid = ", os.getpid())

    pickle.dump(extracted_features, open(save_file, 'wb+'))


def get_radon_transform(img, n_beams=64):
    # Scale image for desired beam count
    scale = n_beams / np.ceil(np.sqrt(2) * max(img.shape))
    img = rescale(img, scale=scale, mode="reflect")

    sinogram = radon(img, theta=np.arange(360), circle=False)

    return sinogram.flatten()
