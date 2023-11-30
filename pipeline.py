# Import require libraries
import numpy as np
from preprocessing_functions import apply_median_filter
from preprocessing_functions import apply_gaussian_filter
from segmentation_functions import kmeans_segmentation

def pipeline(img):
    # img = apply_gaussian_filter(img, sigma=2, kernel_size=3)
    img = apply_median_filter(img, kernel_size=3)
    return img