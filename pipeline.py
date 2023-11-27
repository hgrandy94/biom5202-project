# Import require libraries
import numpy as np
from preprocessing_functions import apply_laplace_filter
from preprocessing_functions import apply_gaussian_filter
from segmentation_functions import kmeans_segmentation

def pipeline(img, output_file_path):
    img = apply_gaussian_filter(img, sigma=2, kernel_size=3)
    img = kmeans_segmentation(img, k=3)
    
    np.save(output_file_path, img)