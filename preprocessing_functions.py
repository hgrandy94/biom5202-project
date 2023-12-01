# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, exposure

# Define functions to perform general preprocessing steps
# NOTE: Approach and Default values obtained form yashbhalgat MRNet-Competition on GitHub
def add_padding(img, input_dim=224):
    # Calculate the amount of padding needed for the view 
    pad = int((img.shape[2] - input_dim)/2)
    # Apply padding to the view
    padded_img = img[:,pad:-pad,pad:-pad]
    return padded_img

def normalize_img(img, max_pixel_val=255):
    # Normalize intensity values so they fall within the range [0, max_pixel_val]
    normalized_img = (img-np.min(img))/(np.max(img)-np.min(img))*max_pixel_val
    return normalized_img

def standardize_img(img, mean=58.09, std=49.73):
    # Standardize pixel values by subracting the mean and dividing by std dev
    standardized_img = (img - mean) / std
    return standardized_img

def create_three_channel_img(img):
    # Create a 3-channel image from the single-channel image
    # This is done to match the input format expected by the model
    three_channel_img = np.stack((img,)*3, axis=1)
    return three_channel_img

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

def canny_edge(img, sigma=3):
    edge_canny = feature.canny(img.astype('float32'), sigma=sigma) # this outputs a boolean array
    edge_canny = edge_canny.astype(np.uint8) * 255 # convert to uint8
    return edge_canny

def histogram_equalization(img):
    hist_eq = exposure.equalize_hist(img)
    hist_eq = hist_eq.astype(np.uint8)
    return hist_eq
