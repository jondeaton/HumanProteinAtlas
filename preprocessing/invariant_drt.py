'''
File:        Invariant Discrete Radon Transform (DRT)
Date:        11/16/18
Authors:     Robert Neff (rneff@stanford.edu)
Description: Implements the discrete radon transformation with rotational, 
             shift and scaling invariance. Note: image must be grayscale!
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from skimage.transform import rescale
from skimage.transform._warps_cy import _warp_fast

'''
Function: compute_drt
---------------------
Computes the discrete radon transform of an image given a set of angles,
at which to take beam sums, and the number of beams per angle.

Ref: https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/radon_transform.py
'''
def compute_drt(img, angles=None, n_beams=600):
    if (img.ndim != 2):
        raise ValueError("Error: Input image must be 2D")
    if (angles is None):
        angles = np.arange(360) # full 360 degrees for rotational invariance

    # Scale image for desired beam count
    scale = n_beams / np.ceil(np.sqrt(2) * max(img.shape))
    img = rescale(img, scale=scale, mode="reflect")
    
    # Pad image (same as radon_transform.py)
    diagonal = np.sqrt(2) * max(img.shape)
    pad = [int(np.ceil(diagonal - s)) for s in img.shape]
    new_center = [(s + p) // 2 for s, p in zip(img.shape, pad)]
    old_center = [s // 2 for s in img.shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(img, pad_width, mode="constant", constant_values=0)

    assert padded_image.shape[0] == padded_image.shape[1] # assert padded image is square
    
    # Define radon image vars
    sinogram = np.zeros((padded_image.shape[0], len(angles)))
    center = padded_image.shape[0] // 2

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    # Construct matrix for rotating (beam directions)
    def build_rotation(angle):
        T = np.deg2rad(angle)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    # Build sinogram
    for i in range(len(angles)):
        rotated_img = _warp_fast(padded_image, build_rotation(angles[i]))

        # Compute ith angle beam sum vector, i.e. all beams on current angle
        sinogram[:, i] = rotated_img.sum(0)
        
    return sinogram

'''
Function: interp_resize_1d
--------------------------
Interpolates a 1D array to a new size.
'''
def interp_resize_1d(arr, new_len):
    arr_interp = interp.interp1d(np.arange(arr.size), arr)
    return arr_interp(np.linspace(0, arr.size - 1, new_len))

'''
Function: decimate_zeros
------------------------
Decimates the zero values from a provided vector (i.e. only keeps non-zero components),
then rescales the resulting vector to the provided size via interpolation. This
ensures there is shift invariance.
'''
def decimate_zeros(proj, new_size):
    decimated_proj = []
    
    # Get rid of zero values
    for i in range(len(proj)):
        if (proj[i] != 0):
            decimated_proj.append(proj[i])

    # Interpolate across desired new size
    return interp_resize_1d(np.array([decimated_proj]), new_size)

'''
Function: post_process_sinogram
-------------------------------
Normalizes the sinogram at each angle and decimates zero values within those
columns, rescaling the vectors to a provided size via interpolation. This ensures
there is scale invariance.
'''
def post_process_sinogram(sinogram, scaled_size=400):
    if (scaled_size == None):
        scaled_size = int(sinogram.shape[0] / 2)
	
    rescaled_sinogram = np.zeros((scaled_size, sinogram.shape[1]))
    
    # Loop over projection at each angle
    for i in range(sinogram.shape[1]):
        # Rescale and interpolate by dropping zeros
        rescaled_sinogram[:, i] = decimate_zeros(sinogram[:, i], scaled_size)
            
        # Normalize beam sum vector (*1000 so values not too small)
        rescaled_sinogram[:, i] = rescaled_sinogram[:, i] / np.linalg.norm(rescaled_sinogram[:, i]) * 1000.
        
    return rescaled_sinogram

'''
Function: plot_sinogram_angles
------------------------------
Plots the beam sums for a specific set of angles.
'''
def plot_sinogram_angles(sinogram, angles, outpath):
    plt.figure(figsize=(8, 8.5))
    plt.title("DRT signals at specified angles")
    plt.xlabel("Features")
    plt.ylabel("Signal")

    lines = []
    for i in angles:
        line_i, = plt.plot(sinogram[:, i], label="Angle %d" % i)
        lines.append(line_i)
    plt.legend(handles=lines)
    plt.savefig(outpath + "sinogram_angles_plot.png")
    
'''
Function: plot_sinogram
-----------------------
Plots the provided sinogram alongside the original image.
'''
def plot_sinogram(sinogram, img, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    
    ax1.set_title("Original Image")
    ax1.imshow(img, cmap=plt.cm.Greys_r)
    
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection beam (jth beam sum)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, sinogram.shape[1], 0, sinogram.shape[0]), aspect="auto")
    fig.tight_layout()
    
    plt.savefig(outpath + "sinogram_plot.png")