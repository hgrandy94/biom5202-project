### BIOM 5202 Final Project
#### Classifying Knee Injuries
##### By: Phil Forster & Heather Grandy

Image Processing Scripts

- preprocessing_functions.py: This script contains four image preprocessing steps to be used in our image processing pipeline. When executing the image processing pipeline (data_loading.py), functions within this script are called by the pipeline function.
    
    apply_gaussian_filter: Applies a Gaussian filter to the input image. Sigma and kernel size must be  specified as input parameters.

    apply_median_filter: Applies a median filter to the input image. Kernel size must be specified as
    an input parameter.

    canny_edge: Performs Canny edge detection on the input image. Sigma, the standard deviation for the Gaussian distribution used as part of the Canny edge detection process must be specified.

    histogram_equalization: Applies histogram equalization to the input image.

