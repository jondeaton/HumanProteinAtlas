#!/usr/bin/env python
"""
File: predict
Date: 12/06/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from HumanProteinAtlas import Dataset
# from feature_extraction import extract_features


def predict_sample_prob(gmm_model, pca_model, human_protein_atlas, ids, save_dir="", X=None):
	pass

	# TODO: update to make predicting test cluster proabilities easy for use in model

    # assert isinstance(model, GaussianMixture)
    # assert isinstance(ids, list)
    # assert isinstance(human_protein_atlas, Dataset)

    # # Extract
    # if X == None:
    #     X = extract_features(human_protein_atlas, ids, save_dir=save_dir)


    # # TODO: setup as needed in CNN

    # y_probs = model.predict_proba(x)

    # return y_probs





