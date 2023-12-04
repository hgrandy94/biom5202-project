# Import required libraries
import numpy as np
from preprocessing_functions import apply_median_filter
from preprocessing_functions import apply_gaussian_filter
from preprocessing_functions import canny_edge, histogram_equalization
from segment import segment_image
import joblib

# Setup variables for segmentation
offsets = [1, 3]
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
properties = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# Load in previously trained model
clf = joblib.load("models/model_20231204140410.joblib")
# Set patch size (required trial and error)
patch_size = 16

def pipeline(img):
    # Pipeline run 2
    # img = apply_gaussian_filter(img, sigma=2, kernel_size=3)
    # Pipeline run 3
    # img = apply_median_filter(img, kernel_size=3)
    # Pipeline run 4
    #img = apply_median_filter(img, kernel_size=9)
    # Pipeline run 5
    #img = canny_edge(img, sigma=3)
    #Pipeline run 6
    #img = apply_median_filter(img, kernel_size=3)
    #img = canny_edge(img, sigma=3)
    # Pipeline run 7
    # alt Canny edge detection
    # Pipeline run 8
    #img = histogram_equalization(img)
    # Pipeline run 9
    segmented_img = segment_image(img, patch_size, offsets, angles, properties, clf)

    return img, segmented_img