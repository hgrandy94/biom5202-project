# Import required libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
from segmentation_train import calculate_gray_level_comatrix_stats


def segment_image(img, patch_size, offsets, angles, properties, svm_model):
    """
    This function segments an image based on its GLCM features as inputs to an SVM model.

    Inputs
    ------
    img : numpy array
        Image to be segmented
    
    patch_size : int
        Should be the same as the patch size used in prepare_patches.py
        Used to split up image into "patches" for GLCM
    
    offsets : list
        Defines distances/offsets between neighbours

    angles : list
        Angles to use in graycomatrix function

    properties : list
        List of valid features/properties to include in the GLCM statistics.
        For example: homogeneity, contrast, etc.
    
    svm_model : sklearn classifier
        SVM classifier for segmentation
    
    Output
    -------
    segmented_image : numpy array
        Resulting segmented image  
    
    """
    # Calculate the number of chunks in each dimension
    num_patch_x = (img.shape[1] + patch_size - 1) // patch_size
    num_patch_y = (img.shape[0] + patch_size - 1) // patch_size

    # Pad the image so that it can be split into chunks evenly
    padded_image = np.pad(
        img,
        ((0, num_patch_y * patch_size - img.shape[0]),
        (0, num_patch_x * patch_size - img.shape[1])), mode="reflect")

    # Split the image into chunks
    chunks = []
    for i in range(num_patch_y):
        for j in range(num_patch_x):
            chunk = padded_image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size]
            chunks.append(chunk)

    # Calculate gray level co-occurrence matrix statistics for each chunk
    features = []
    for chunk in chunks:
        stats = calculate_gray_level_comatrix_stats(chunk, offsets, angles, properties)
        features.append(stats)

    # Pass the features into the trained SVM model for segmentation
    predictions = svm_model.predict(features)

    # Return the segmented image
    segmented_image = np.zeros_like(img)
    index = 0
    for i in range(num_patch_y):
        for j in range(num_patch_x):
            segmented_image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ] = predictions[index]
            index += 1

    return segmented_image

# MAIN SCRIPT
if __name__=="__main__":
    offsets = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    # Load in previously trained model
    clf = joblib.load("models/model_20231204140410.joblib")
    # Set chunk size (required trial and error)
    patch_size = 16

    # Let's test one image slice, selected arbitrarily
    image = np.load("C:/Users/heath/projects/uOttawa/classes/MRNet-v1.0/train/sagittal/0099.npy")
    image_slice = image[9]

    # Segment the image using the segment_image function
    segmented_image = segment_image(image_slice, patch_size, offsets, angles, properties, clf)

    # Plot the results!!!
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_slice, cmap="gray")
    ax[1].imshow(segmented_image, cmap="gray")
    plt.show()

    # Plot as contours overlayed on the original image
    plt.imshow(image_slice, cmap="gray")
    plt.contour(segmented_image, levels=[0.5], colors="red")
    plt.show()
