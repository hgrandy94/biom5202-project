# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage

# Define function to compute mean squared error
# this function can compute MSE for any two given image arrays
def compute_mse(img1, img2):
    """
    This function computes the mean squared error of two arrays

    Parameters
    ------------
    img1, img2: 2D numpy array
        Arrays for which the mse will be computed
    
    Output
    ------------
    mse: float
        The mean squared error of the two arrays
    """
    mse = np.square(img1 - img2).mean()
    return mse

# Specify linear Gaussian filter 
def apply_gaussian_filter(img, sigma, kernel_size):
    """
    This function applies a Gaussian filter to a data array and was designed
    with the intent of filtering image arrays. The resulting Gaussian-filtered image is displayed.

    Parameters
    ------------
    img: 2D numpy array
        Array which the Gaussian filter will be applied to (the image array
        with noise applied).
    sigma: integer
        Sigma (standard deviation) for the Gaussian distribution used for filtering.
    kernel_size: integer
        Desired kernel/mask size for the Gaussian filter.        
    
    Output
    ------------
    N/A - All computations done within function
    """
    # Convert kernel to radius as required by ndimage.gaussian_filter
    radius = kernel_size // 2
    # Apply gaussian filter
    filtered_img = ndimage.gaussian_filter(img, sigma, radius=radius)

    # Display the filtered image
    #plt.imshow(filtered_img, cmap="Greys_r")
    #plt.title(f"{kernel_size}x{kernel_size} Gaussian Filter (sigma={sigma})")

    return filtered_img

def apply_median_filter(img, kernel_size):
    """
    This function applies a median filter to a data array and was designed
    with the intent of filtering image arrays. The resulting median-filtered image is displayed.

    Parameters
    ------------
    img: 2D numpy array
        Array which the Gaussian filter will be applied to (the image array
        with noise applied).
    kernel_size: integer
        Desired kernel/mask size for the Gaussian filter.       
    
    Output
    ------------
    N/A - All computations done within function
    """
    # Apply median filter
    filtered_img = ndimage.median_filter(img, (kernel_size, kernel_size))

    # Display the filtered image
    # plt.imshow(filtered_img, cmap="Greys_r")
    # plt.title(f"{kernel_size}x{kernel_size} Median Filter")
    
    return filtered_img

def apply_laplace_filter(input_img, k, kernel_size=9):
    """
    This function applies a Laplacian filter to a data array and was designed
    with the intent of filtering image arrays. It clips the laplace values to adjust
    results to a valid scale, and finally plots the Laplacian-filtered image.

    Parameters
    ------------
    input_img: 2D numpy array
        Array to which the Laplacian filter will be applied (image array)
    k: integer
        Scaling factor to adjust weight of edges.
    kernel_size: integer
        Kernel size for the Laplacian transform. Default=9 if not specified

    Output
    ------------
    laplace_clipped: 2D numpy array
        Image array with Laplacian transform and clipping applied.
    """
    # Setup Laplace filter using skimage library
    laplace_filter = skimage.filters.laplace(input_img, ksize=kernel_size)

    # Apply the Laplace filter to the input image
    apply_laplace = input_img + k * laplace_filter

    # Adjust clipping to valid scale
    # laplace_clipped =apply_laplace
    laplace_clipped = np.clip(apply_laplace, 0, 1)

    # Plot the Laplacian-enhanced knee x-ray image
    #fig, ax = plt.subplots(1,1)
    ##ax[0].imshow(knee_array_scaled, cmap="Greys_r")
    #ax.imshow(laplace_clipped, cmap="Greys_r")
    #ax[0].set_title("Original Image")
    #ax.set_title(f"Image with Laplacian Filter, k={k}")
    #plt.tight_layout();
    return laplace_clipped