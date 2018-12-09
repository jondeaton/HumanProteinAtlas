#!/usr/bin/env python
"""
File: preprocess
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from scipy.signal import medfilt

def preprocess_dataset(dataset):
    # currently no universal pre-processing is necessary
    return dataset


def get_binary_image(image, threshold=0.5):
	b_img = np.empty_like(image)

	for layer in range(len(image)):
		b_img[layer] = medfilt(image[layer], 1)

		b_img[layer][b_img[layer] < threshold] = 0.0
		b_img[layer][b_img[layer] >= threshold] = 1.0

	return b_img


def get_grayscale_image(image):
	return (image[0] + image[1] + image[2]) / 3