#!/usr/bin/env python
"""
File: __init__.py
Date: 11/25/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from enum import Enum
import numpy as np

from skimage.feature import greycomatrix, greycoprops, ORB, local_binary_pattern, hog
from skimage.filters import gabor_kernel
from skimage.transform import rescale
from scipy import ndimage as ndi
from scipy.fftpack import dct

from feature_extraction.drt import get_radon_transform
import feature_extraction.feature_visualization
import feature_extraction.radonCDT as RCDT
from preprocessing.preprocess import get_binary_image, get_grayscale_image


class Feature(Enum):
    drt = 1
    surf = 2
    dct = 3
    lbp = 4
    orb = 5
    hog = 6
    gabor = 7
    drt_cdt = 8


def extract_features(images, method=Feature.drt):
    features = list()
    for image in images:
        features.append(get_features(image, method))
    return np.concatenate(features, axis=0)


def get_features(image, method=Feature.drt):
    """
    TODO: add additional arguments ...

    :param image:
    :param method:
    :return:
    """

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
    elif method == Feature.gabor:
        return get_gabor_features(image)
    elif method == Feature.drt_cdt:
        return get_radon_cdt_features(image)
    else:
        raise Exception("Unhandled feature extraction method.")


"""
Get DRT (discrete radon transform) features.
"""
def get_radon_features(image):
    image = get_grayscale_image(image)
    image = get_binary_image(image)

    return get_radon_transform(image).flatten()


def get_radon_cdt_features(image):
    features = []

    image = get_binary_image(image)

    rcdt=RCDT.RadonCDT()

    for layer in range(len(image)):
        scale = 64 / np.ceil(np.sqrt(2) * max(image[layer].shape))
        img = rescale(image[layer], scale=scale, mode="reflect")
        features.append(rcdt.transform(img).flatten())

    return np.concatenate(features)


"""
Get SURF (Speeded-Up Robust Features, i.e. fast SIFT) features.

Lower hessian thershold results in more features.
"""
def get_surf_features(image, hessian_threshold=500):
    raise Exception("Not yet implemented.")
    # surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # kp, des = surf.detectAndCompute(image[0] / 255, None)

    # print(len(key_points))


"""
Get block based ULC (upper left corner) DCT features.

n_block_features must be a square number.

Resulting features size is: (512 / block_size)^2 * block_features_width^2 * 3
"""
def get_dct_features(image, block_size=512, block_features_width=20):
    n_blocks = int(image.shape[1] / block_size)
    features = []

    image = get_grayscale_image(image)
    # image = get_binary_image(image)

    for i in range(n_blocks):
        for j in range(n_blocks):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

            block_features = dct(block, norm="ortho")[:block_features_width, :block_features_width].flatten()

            features.append(block_features)

    return np.concatenate(features)


"""
Get histogram of Local Binary Patterns of image.

Resulting features size is: neighbors * radius

Note: gray-scale and rotation invariant. Should also
be translation invariant if neighbor radius is small.
"""
def get_local_binary_patterns(image, neighbors=14, radius=4):
    numPoints = neighbors * radius

    features = []

    for layer in range(len(image)):
        lbp_image = local_binary_pattern(image, numPoints, radius, method="uniform")
        hist = get_lbp_histogram(lbp_image)

        # feature_visualization.show_lbp_features(lbp_image)

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

Resulting features size is: n_keypoints * 2 (maybe)
"""
def get_orb_features(image, n_keypoints=5, patch_size=20):
    features = []

    for layer in range(len(image)): # loop over image layers
        key_points= get_orb_keypoints(image[layer], n_keypoints)

        if len(key_points) != n_keypoints:
            key_points = np.concatenate((key_points, [[256, 256]] * (n_keypoints - len(key_points))))

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


def get_orb_keypoints(image, n_keypoints):
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05, harris_k=0.1)
    orb.detect(image)

    return orb.keypoints


"""
Get HoG (Histogram of Gradients) features.

Smaller pixels_per_cell, larger cells_per_block increases features size.
"""
def get_hog_features(image):
    reshaped_image = np.swapaxes(image, 0, 2)

    features = hog(reshaped_image, orientations=10, pixels_per_cell=(16, 16), cells_per_block=(1, 1), multichannel=True)

    # feature_visualization.show_hog_features(reshaped_image)

    return features

"""
Get Gabor filter/kernel features.

Captures magnitude at different orientations and scales. Good at edge detection.

Check this out: https://stackoverflow.com/questions/20608458/gabor-feature-extraction

Probably pass on this one.
"""
def get_gabor_features(image, frequency=0.2, theta=0, sigma_x=1, sigma_y=1):

    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y))
    filtered = ndi.convolve(image, kernel, mode='wrap')

    # TODO: decide if going to try this
    # need more kernels and to choose some features out of huge set

    # plt.imshow(filtered)
    # plt.show()

    raise Exception("Not yet implemented.")
