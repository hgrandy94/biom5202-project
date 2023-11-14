# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pipeline import pipeline

# set file path
file_path = "input_images/sagittal-0011-slice12.png"
# load image
img = cv2.imread(file_path)

# Set output directory
output_images = "output_images"

# Let's setup a test
output_img = pipeline(img)
# Save resulting image
cv2.imwrite(f"{output_images}/test_image.png", output_img)