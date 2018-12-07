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
from sklearn.decomposition import IncrementalPCA, PCA

from HumanProteinAtlas import Dataset, Sample


def extract_pca_features(pca_model, X, save_dir="", training=True):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    pca_features_file = os.path.join(save_dir, "pca_features_" + str(d) + "_train" if training else "_test")
    if os.path.exists(pca_features_file):
        with open(pca_features_file, "rb" ) as file:
            return pickle.load(file)

    pca_features = pca_model.transform(X)
 
    if save_dir != "":
        with open(pca_features, 'wb+') as file:
            pickle.dump(pca_features, file)

    return pca_features


def batch_disk_pca_transform(pca_model, save_dir, training=True):
    pca_features_file = os.path.join(save_dir, "pca_features_" + str(d) + "_train" if training else "_test")
    if os.path.exists(pca_features_file):
        with open(pca_features_file, "rb" ) as file:
            return pickle.load(file)

    outfile_names = compute_radon_features(human_protein_atlas, ids, save_dir, training=training)

    # Batch transform and save features
    pca_features = []
    for i, name in enumerate(outfile_names):
        features = None
        with open(name, "rb" ) as file:
            features = np.array(pickle.load(file))

        transformed = pca_model.transform(features)
        if i == 0:
            pca_features = transformed
        else:
            pca_features = np.concatenate((pca_features, transformed), axis=0)

    # Save the transformed features
    with open(pca_features_file, 'wb+') as file:
        pickle.dump(pca_features, file)

    return pca_features


def full_fit_pca_model(human_protein_atlas, ids, d=0.95, save_dir=""):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    # Check if model saved
    pca_model_file = os.path.join(save_dir, "pca_model_full"  + str(d))
    if os.path.exists(pca_model_file):
        with open(pca_model_file, "rb" ) as file:
            return pickle.load(file)

    # Get batch radon saved files
    outfile_names = compute_radon_features(human_protein_atlas, ids, save_dir)

    pca_model = PCA(n_components=d, whiten=True)

    # Load features into one file
    features = None
    for i, name in enumerate(outfile_names):
        batch_features = None
        with open(name, "rb" ) as file:
            batch_features = np.array(pickle.load(file))

        if i == 0:
            features = batch_features
        else:
            features = np.concatenate((batch_features), axis=0)

    pca_model.fit(features)

    # Save the fitted model
    with open(pca_model_file, 'wb+') as file:
        pickle.dump(pca_model_file, file)

    return pca_model


def batch_fit_pca_model(human_protein_atlas, ids, d=1000, save_dir=""):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    # Check if model saved
    pca_model_file = os.path.join(save_dir, "pca_model_batch"  + str(d))
    if os.path.exists(pca_model_file):
        with open(pca_model_file, "rb" ) as file:
            return pickle.load(file)

    # Get batch radon saved files
    outfile_names = compute_radon_features(human_protein_atlas, ids, save_dir)

    pca_model = IncrementalPCA(n_components=d, whiten=True)

    # Batch fit new PCA model
    for name in outfile_names:
        features = None
        with open(name, "rb" ) as file:
            features = pickle.load(file)

        pca_model.partial_fit(features)

    # Save the fitted model
    with open(pca_model_file, 'wb+') as file:
        pickle.dump(pca_model_file, file)

    return pca_model


def compute_radon_features(human_protein_atlas, ids, save_dir="", sequential=True, memory=False, training=True):
    assert isinstance(ids, list)
    assert isinstance(human_protein_atlas, Dataset)

    # Read features directly into memory
    if memory: 
        return np.array(radon_features_helper(human_protein_atlas, ids))

    # Write features to split files on disk, return filenames
    if sequential:
        return sequential_batch_disk_radon_features(human_protein_atlas, ids, save_dir, training)
    else:
        return multiprocessed_batch_disk_radon_features(human_protein_atlas, ids, save_dir, training)


def sequential_batch_disk_radon_features(human_protein_atlas, ids, save_dir, batch_size=10, training=True):
    for i, id_subset in enumerate(np.array_split(ids, batch_size)):
        filename = os.path.join(save_dir, "radon_features_" + str(i) + ("_train" if training else "_test") + ".radon_data")

        if os.path.exists(filename): # file already exists, don't recompute
            continue

        radon_features_helper(human_protein_atlas, id_subset, filename, log_progress=True)

    outfile_names = []

    # Verify all files created
    for i in range(batch_size):
        filename  = os.path.join(save_dir, "radon_features_" + str(i) + ("_train" if training else "_test") + ".radon_data")
        
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0

        outfile_names.append(filename)

    return outfile_names


def multiprocessed_batch_disk_radon_features(human_protein_atlas, ids, save_dir, batch_size=30, max_processes=2, training=True):
    # Get CPU count
    try: 
        n_cpus = len(os.sched_getaffinity(0)) # useable cpu count, should work on Unix-base
    except:
        n_cpus = mp.cpu_count() # total cpu count, works on Windows

    # Spawn process to handle subset of ids extraction
    processes = []
    for i, id_subset in enumerate(np.array_split(ids, batch_size)):
        filename = os.path.join(save_dir, "radon_features_" + str(i) + ("_train" if training else "_test") + ".radon_data")

        if os.path.exists(filename): # file already exists, don't recompute
            continue
        
        p = mp.Process(target=radon_features_helper, args=(human_protein_atlas, id_subset, filename))
        processes.append(p)
        p.start()

        if len(processes) % min(n_cpus, max_processes) == 0: # wait for oldest process to finish before spawning a new one
            processes[0].join()
            processes.pop(0)
    
    # Wait for all remaining processes to finish
    for p in processes:
        p.join()

    outfile_names = []

    # Verify all files created
    for i in range(batch_size):
        filename  = os.path.join(save_dir, "radon_features_" + str(i) + ("_train" if training else "_test") + ".radon_data")
        
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0

        outfile_names.append(filename)

    return outfile_names


def radon_features_helper(human_protein_atlas, ids, save_file="", log_progress=False):
    assert isinstance(human_protein_atlas, Dataset)

    if os.path.exists(save_file):
        with open(save_file, "rb" ) as file:
            return pickle.load(file)

    extracted_features = []

    for i, id in enumerate(ids):
        sample = human_protein_atlas[id]
        assert isinstance(sample, Sample)

        yellow_features = get_radon_transform(sample.yellow)
        red_features = get_radon_transform(sample.red)
        blue_features = get_radon_transform(sample.blue)
        sample_features = np.concatenate((yellow_features, red_features, blue_features))

        extracted_features.append(sample_features)

        if log_progress and i % 100 == 0: 
            print("Progress tick, processed images = ", i, "pid = ", os.getpid())

    if save_file != "":
        with open(save_file, 'wb+') as file:
            pickle.dump(extracted_features, file)
    
    return extracted_features


def get_radon_transform(img, n_beams=64, angle_step=3):
    # Scale image for desired beam count
    scale = n_beams / np.ceil(np.sqrt(2) * max(img.shape))
    img = rescale(img, scale=scale, mode="reflect")

    sinogram = radon(img, theta=np.arange(0, 360, angle_step), circle=False)

    return sinogram.flatten()
