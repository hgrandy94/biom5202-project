# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
from pipeline import pipeline
import time
from multiprocessing import Pool

def process_file(input):
    """
    This function ensures only sagittal view images are passed to the image processing
    pipeline for segmentation. The resulting images are added to a numpy array in the format
    required for model training.

    Inputs
    -------
    input : tuple
        Contains the absolute file path of the input files and the output path
        where the processed images will be stored.
    
    Outputs
    --------
    Nothing is returned, but the processed images are saved in the specified output directory.
    """
    abs_file, output_file_path = input
    if "sagittal" in abs_file:
        # If the file is in .npy format, load the data and process each slice
        image_data = np.load(abs_file)
        
        # Iterate through each slice
        # Initialize list to track output slices
        output_slices = []
        for i, image_slice in enumerate(image_data):
            # Store both the original image slice and the segmented slice
            original_slice, segmented_slice = pipeline(image_slice)
            # Stack both the original and segmented slice that will work with TripleMRNet
            output_slice = np.stack([original_slice, segmented_slice], axis=-1)
            output_slices.append(output_slice)
            #print(f"Processed and saved slice {i} from .npy: {file_path} to {output_file_path}")
        
        # Save in npy format
        output_slices_array = np.array(output_slices)
        np.save(output_file_path, output_slices_array)

    else:
        # If image is not sagittal, just copy it
        shutil.copy(abs_file, output_file_path)


if __name__=="__main__":
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

    # KEEP TRACK OF TIME STARTING FROM THIS POINT
    start_time = time.time()

    # Initialize array that will store which files need to be processed
    files_to_process =[]

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

            output_file_path = os.path.join(output_subdirectory, file)

            # Check if the file is a CSV
            if file.endswith('.csv'):
                shutil.copy(file_path, output_file_path)
                #print(f"Copied CSV: {file_path} to {output_file_path}")
            elif file.endswith('.npy'):
                # Append to files_to_process list
                files_to_process.append((file_path,output_file_path))

            else:
                # Handle other file types (catch-all case)
                print(f"Ignored file: {file_path} (unsupported file type)")

    # Process files using Python multiprocessing functionality
    with Pool(5) as p:
        p.map(process_file, files_to_process)

    print("Iterating done! Images successfully saved.")
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Script execution time: {elapsed_time} seconds")