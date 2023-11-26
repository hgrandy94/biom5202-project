# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import os
from pipeline import pipeline

# Input directory pointing to the .npy files
input_dir = "C:\\Users\\heath\\OneDrive - University of Ottawa\\Courses\\BIOM5202\\MRNet-v1.0\\MRNet-v1.0"

# Set output directory
output_dir = "C:\\Users\\heath\\OneDrive - University of Ottawa\\Courses\\BIOM5202\\Project\\output_images"

# Set dynamic run directory
# Get the current date and time
now = datetime.now()
# Format the date and time as a string
timestamp_str = now.strftime("%Y%m%d_%H%M%S")
# Use an f-string to create the run directory with the timestamp
run_dir = f"run_{timestamp_str}"
run_dir = os.path.join(output_dir, run_dir)

# CREATE DIRECTORY
# Check if the directory exists
if not os.path.exists(run_dir):
    # Create the directory if it doesn't exist
    os.makedirs(run_dir)
    print(f"Directory '{run_dir}' created.")
else:
    print(f"Directory '{run_dir}' already exists.")

# Iterate through all files recursively in the input directory
