# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, exposure, img_as_ubyte

# Define functions to perform general preprocessing steps
# Specify linear Gaussian filter 
def apply_gaussian_filter(img, sigma, kernel_size):
    """
    This function applies a Gaussian filter to a data array and was designed
    with the intent of filtering image arrays. The resulting Gaussian-filtered image is displayed.

    Parameters
    ------------
    img: 2D numpy array
        Array which the Gaussian filter will be applied to.

    sigma: integer
        Sigma (standard deviation) for the Gaussian distribution used for filtering.

    kernel_size: integer
        Desired kernel/mask size for the Gaussian filter.        
    
    Output
    ------------
    filtered_img : 2D numpy array
        Image array after Gaussian filter is applied
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
    with the intent of filtering image arrays. The resulting median-filtered image is returned.

    Parameters
    ------------
    img: 2D numpy array
        Array which the median filter will be applied to.

    kernel_size: integer
        Desired kernel/mask size for the Gaussian filter.       
    
    Output
    ------------
    filtered_img : 2D numpy array
        Image array after median filter is applied
    """
    # Apply median filter
    filtered_img = ndimage.median_filter(img, (kernel_size, kernel_size))

    # Display the filtered image
    # plt.imshow(filtered_img, cmap="Greys_r")
    # plt.title(f"{kernel_size}x{kernel_size} Median Filter")
    
    return filtered_img

def canny_edge(img, sigma=3):
    """
    This function performs Canny edge detection on the input image and returns the
    Canny edge map.

    Parameters
    ------------
    img: 2D numpy array
        Image which the Canny edge detector will be applied to.

    sigma: integer
        Sigma (standard deviation) for the Gaussian distribution used for filtering. This is part of
        the Canny edge detection process.        
    
    Output
    ------------
    edge_canny : 2D numpy array (uint8)
        Canny edge map of input image
    """
    # Use skimage functionality to perform Canny edge detection
    edge_canny = feature.canny(img.astype('float32'), sigma=sigma) # this outputs a boolean array
    edge_canny = edge_canny.astype(np.uint8) * 255 # convert to uint8
    return edge_canny

def histogram_equalization(img):
    """
    This function performs histogram equalization on the input image and returns the
    resulting image

    Parameters
    ------------
    img: 2D numpy array
        Image which histogram equalization will be applied to.
    
    Output
    ------------
    hist_eq : 2D numpy array (uint8)
        Image array after histogram equalization
    """
    # Use skimage functionality to perform histogram equalization
    hist_eq = exposure.equalize_hist(img)
    hist_eq = img_as_ubyte(hist_eq) # convert to uint8
    return hist_eq
