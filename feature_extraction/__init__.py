#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from enum import Enum
import numpy as np

from skimage.feature import greycomatrix, greycoprops, ORB, local_binary_pattern, hog
from scipy.fftpack import dct
import cv2

from feature_extraction.drt import get_radon_transform
import feature_extraction.feature_visualization


class Feature(Enum):
    drt = 1
    surf = 2
    dct = 3
    lbp = 4
    orb = 5
    hog = 6


def extract_features(images, method=Feature.drt):
    features = list()
    for image in images:
        features.append(get_features(image, method))
    return np.concatenate(features, axis=0)


def get_features(image, method=Feature.drt): # TODO: add additional arguments ...
    if method == Feature.drt:
        return get_radon_features(image)
    elif method == Feature.surf:
        return get_surf_features(image)
    elif method == Feature.dct:
        return get_dct_features(image)
    elif method == Feature.lbp:
        return get_local_binary_patterns(image)
    elif method == Feature.orb:
        return get_orb_features(image)
    elif method == Feature.hog:
        return get_hog_features(image)
    else:
        raise Exception("Unhandled feature extraction method.")


"""
Get DRT (discrete radon transform) features.
"""
def get_radon_features(image):
    drt = get_radon_transform(image)
    return drt


"""
Get SURF (Speeded-Up Robust Features, i.e. fast SIFT) features.

Lower hessian thershold results in more features.
"""
def get_surf_features(image, hessian_threshold=500):
    pass

    # TODO: not working, FIXME!

    # surf = cv2.SURF(n_components)
    # key_points, descriptors = surf.detectAndCompute(image, None)


"""
Get block based ULC (upper left corner) DCT features.

n_block_features must be a square number.

Resulting features size is: (512 / block_size)^2 * block_features_width^2 * 3
"""
def get_dct_features(image, block_size=256, block_features_width=8):
    n_blocks = int(image.shape[1] / block_size)

    features = []

    for layer in range(image.shape[0]):
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = image[layer, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

                block_features = dct(block, norm="ortho")[:block_features_width, :block_features_width].flatten()

                features.append(block_features)

    return np.concatenate(features)


"""
Get histogram of Local Binary Patterns of image.

Resulting features size is: neighbors * radius

Note: gray-scale and rotation invariant. Should also
be translation invariant if neighbor radius is small.
"""
def get_local_binary_patterns(image, neighbors=8, radius=3):
    numPoints = neighbors * radius

    features = []

    for layer in range(len(image)):
        lbp_image = local_binary_pattern(image[layer], numPoints, radius, method="uniform")
        hist = get_lbp_histogram(lbp_image)

        # feature_visualization.show_lbp_features(lbp)

        features.append(hist)

    return np.concatenate(features)
    

def get_lbp_histogram(lbp_image):
    nbins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image.ravel(), bins=nbins, range=(0, nbins))

    # Normalize
    hist = hist.astype(np.float32)
    hist /= hist.sum()

    return hist


"""
Get ORB (Oriented FAST and rotated BRIEF) keypoint features.

Resulting features size is: TODO
"""
def get_orb_features(image, n_keypoints=500, patch_size=20):
    features = []

    for layer in range(len(image)): # loop over image layers
        key_points, _ = get_keypoints(image[layer], n_keypoints)

        diss = list()
        corr = list()

        for keypoint in key_points:
            x = int(keypoint[0])
            y = int(keypoint[1])
            patch = image[layer, x:x+patch_size, y:y+patch_size].astype(int)

            glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)

            diss.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            corr.append(greycoprops(glcm, 'correlation')[0, 0])

        features.append(np.array(diss + corr))

    return np.concatenate(features)


def get_keypoints(image, n_keypoints):
    orb = ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(image)

    return orb.keypoints, orb.descriptors # TODO: maybe use descriptors?


"""
Get HoG (Histogram of Gradients) features.

Lower hessian thershold results in more features.
"""
def get_hog_features(image):
    pass

    # reshape

    # TODO: in progress, rnef fworking here

    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
    #                 cells_per_block=(1, 1), visualize=True, multichannel=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()