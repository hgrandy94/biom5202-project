# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
from pipeline import pipeline

# Input directory pointing to the .npy files
input_dir = "C:\\Users\\heath\\projects\\uOttawa\\classes\\MRNet-v1.0"

# Set output directory
output_dir = "C:\\Users\\heath\\projects\\uOttawa\\classes\\mrnet-output"

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
# If the file is .csv, copy into output directory
print("Start iterating...")
for root, dirs, files in os.walk(input_dir):
    # Determine the corresponding output subdirectory
    relative_path = os.path.relpath(root, input_dir)
    output_subdirectory = os.path.join(run_dir, relative_path)

    # Create the corresponding output subdirectory if it doesn't exist
    os.makedirs(output_subdirectory, exist_ok=True)

    for file in files:
        file_path = os.path.join(root, file)

        # Check if the file is a CSV
        if file.endswith('.csv'):
            output_file_path = os.path.join(output_subdirectory, file)
            shutil.copy(file_path, output_file_path)
            #print(f"Copied CSV: {file_path} to {output_file_path}")
        elif file.endswith('.npy'):
            # If the file is in .npy format, load the data and process each slice
            image_data = np.load(file_path)
            
            # Iterate through each slice
            # Initialize list to track output slices
            output_slices = []
            for i, image_slice in enumerate(image_data):
                output_file_path = os.path.join(output_subdirectory, f"{file}")
                output_slice = pipeline(image_slice)
                output_slices.append(output_slice)
                #print(f"Processed and saved slice {i} from .npy: {file_path} to {output_file_path}")
            
            # Save in npy format
            output_slices_array = np.array(output_slices)
            np.save(output_file_path, output_slices_array)

        else:
            # Handle other file types (catch-all case)
            print(f"Ignored file: {file_path} (unsupported file type)")

print("Iterating done! Images successfully saved.")

