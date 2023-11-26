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


# Input directory pointing to the .npy files
file_path = ["C:\\Users\\heath\\OneDrive - University of Ottawa\\Courses\\BIOM5202\\MRNet-v1.0\\MRNet-v1.0\\train\\sagittal\\0011.npy"]

# Set output directory
output_images = "output_images"

# Iterate over the .npy files and plot their contents
for file_path in file_path:
    # Load data from the .npy file
    data = np.load(file_path)
    print(np.shape(data))

    # Create a visualization for each slice (assuming 2D slices)
    for slice_index in range(data.shape[0]):
        # Display the slice using Matplotlib
        plt.figure()
        plt.imshow(data[slice_index], cmap='gray')
        plt.title(f'Slice {slice_index + 1} from {file_path}')

        # Optionally, display or save the plot
        # plt.show()  # Display the plot
        # Optionally, save the plot as an image file
        # plt.savefig(f'slice_{slice_index + 1}_{file_path}.png')

        # Close the plot to free up memory
        plt.close()

# # Let's setup a test
# output_img = pipeline(data[27])
# # Save resulting image
# cv2.imwrite(f"{output_images}/test_image.png", output_img)

# Batch loading