#!/usr/bin/env python
"""
File: feature_visualization
Date: 12/07/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import hog
import cv2


"""
Ref: https://aurlabcvsimulator.readthedocs.io/en/latest/PlotLBPsHistogram/
"""
def show_lbp_features(lbp_image):
    # Display as gray-scale image
    cv2.imshow("LBP", lbp_image.astype("uint8"))

    plt.style.use("ggplot")
    (fig, ax) = plt.subplots()
    fig.suptitle("Local Binary Patterns")
    plt.ylabel("% of Pixels")
    plt.xlabel("LBP pixel bucket")

    nbins = int(lbp_image.max() + 1)

    # Plot a histogram of the LBP features
    ax.hist(lbp_image.ravel(), normed=True, bins=nbins, range=(0, nbins))
    ax.set_xlim([0, nbins])
    ax.set_ylim([0, 0.030])
    plt.show()


"""
Ref: http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
"""
def show_hog_features(image):
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, multichannel=True)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()


# TODO: add more feature visualization help, ex. plotting PCA for 2 features