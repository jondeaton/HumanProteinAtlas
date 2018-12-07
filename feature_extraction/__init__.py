#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from enum import Enum
import numpy as np

from feature_extraction.drt import get_radon_transform

from skimage.feature import greycomatrix, greycoprops
import cv2


class Feature(Enum):
    drt_pca = 1
    surf = 2


def extract_features(images):
    features = list()
    for image in images:
        features.append(get_features(image))
    return np.concatenate(features, axis=0)


# def get_features(image):
#     drt = get_radon_transform(image)
#     # todo: something else?
#     return drt

def get_keypoints(unscaled_image):
    scaled_image = np.divide(unscaled_image, 255.0)
    channels_last = np.moveaxis(scaled_image, 0, -1).astype(np.float32)
    orb = cv2.ORB_create(400)
    key_points, description = orb.detectAndCompute(channels_last, None)
    return key_points

def get_features(image, patch_size=20):

    key_points = get_keypoints(image)

    diss = list()
    corr = list()

    for keypoint in key_points:
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        patch = image[x:x+patch_size, y:y+patch_size]
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)

        diss.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        corr.append(greycoprops(glcm, 'correlation')[0, 0])

    return np.array(diss + corr)

