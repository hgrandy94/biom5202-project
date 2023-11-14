# Import require libraries
from preprocessing_functions import apply_laplace_filter
from preprocessing_functions import apply_gaussian_filter

def pipeline(img, og_filename="test", og_slice=20):
    img = apply_gaussian_filter(img, sigma=2, kernel_size=3)
    return img