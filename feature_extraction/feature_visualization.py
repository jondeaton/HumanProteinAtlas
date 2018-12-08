#!/usr/bin/env python
"""
File: feature_visualization
Date: 12/07/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import matplotlib.pyplot as plt
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

# TODO: add more feature visualization help, ex. plotting PCA for 2 features