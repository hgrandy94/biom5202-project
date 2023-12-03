# Import require libraries
import numpy as np
from preprocessing_functions import apply_median_filter
from preprocessing_functions import apply_gaussian_filter
from preprocessing_functions import canny_edge, histogram_equalization
from segmentation_functions import kmeans_segmentation

def pipeline(img):
    # Pipeline run 2
    # img = apply_gaussian_filter(img, sigma=2, kernel_size=3)
    # Pipeline run 3
    # img = apply_median_filter(img, kernel_size=3)
    # Pipeline run 4
    # img = apply_median_filter(img, kernel_size=9)
    # Pipeline run 5
    # img = canny_edge(img, sigma=3)
    #Pipeline run 6
    # img = apply_median_filter(img, kernel_size=3)
    # img = canny_edge(img, sigma=3)
    # Pipeline run 7
    # img = histogram_equalization(img)
    # Pipelin run 8

    return img