# Import required libraries
import joblib
import os
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from skimage.feature import graycomatrix, graycoprops
import tqdm

def calculate_gray_level_comatrix_stats(img, offsets, angles, properties):
    """
    This function computes GLCM feature statstics and returns them in an array.

    Inputs
    -------
    img : numpy array
        Image for which GLCM features will be extracted
    
    offsets : list
        Defines distances/offsets between neighbours
    
    angles: list
        Angles to use in graycomatrix function
    
    properties: list
        List of valid features/properties to include in the GLCM statistics.
        For example: homogeneity, contrast, etc.
    
    Output
    -------
    features : numpy array
        List of GLCM features for specified properties
    
    """
    # Compute gray level co-occurrence matrix
    glcm = graycomatrix(img, offsets, angles, levels=256, symmetric=True, normed=True)

    # Compute gray level co-occurrence matrix properties
    features = [graycoprops(glcm, property) for property in properties]

    # Flatten the features array
    features = np.ravel(features)

    return features


def generate_features(df, folder, offsets, angles, properties):
    """
    This function takes in a dataset (defined by df and folder) and produces
    the GLCM features and labels.

    Inputs
    -------
    df : pandas dataframe
        Contains filepath, slice, x, y positions for points and label to indicate
        whether the position represents the ACL or not.

    folder : string
        Name of the folder where the image patches (.npy) are stored.

    offsets : list
        Defines distances/offsets between neighbours

    angles : list
        Angles to use in graycomatrix function

    properties : list
        List of valid features/properties to include in the GLCM statistics.
        For example: homogeneity, contrast, etc.

    Outputs
    -------
    X : array of arrays
        Contains GLCM features

    y : pandas series
        Labels - ACL (1) and non-ACL (0)
    """

    X_paths = [os.path.join(folder, f"{x:04}.npy") for x in df.index]
    X_patches = [np.load(x) for x in X_paths]

    # Validate that all patches are the same size
    patch_sizes = [x.shape for x in X_patches]
    # Catch any patches that are not the same size
    if len(set(patch_sizes)) != 1:
        print("Patches are not the same size")
        # print the index of the patch that is not the same size
        for i, size in enumerate(patch_sizes):
            if size != patch_sizes[0]:
                print(f"Patch {i} has size {size}")

        sys.exit(1)

    # Calculate features for each image patch
    # Use the previously defined calculate_gray_level_comatrix_stats function
    X = [calculate_gray_level_comatrix_stats(patch, offsets, angles, properties)
        for patch in tqdm.tqdm(X_patches)] # tqdm gives progress bar

    # Get the labels for training
    y = df["label"]

    return X, y

## MAIN SCRIPT
if __name__=="__main__":
    # Setup offsets/neighbours (set by trial and error and to minimize compute time)
    offsets = [1, 3]
    # Setup angles, chosen from referencing skimage docs
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    # Setup GLCM properties list according to what is supported by skimage
    properties = ["contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM"]

    # Set filepaths for the train and valid folders for the segmentation
    train_folder = "segmentation_data/train"
    valid_folder = "segmentation_data/valid"

    # Set filepaths for the train and valid csvs for the image patches
    train_csv = "segmentation_data/img_patches/train.csv"
    valid_csv = "segmentation_data/img_patches/valid.csv"

    # Read in the train and validation csvs as dataframes
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    # Get X and y training data using the generate_features function
    X_train, y_train = generate_features(train_df, train_folder, offsets, angles, properties)

    # Get X and y validationdata using the generate_features function
    X_valid, y_valid = generate_features(valid_df, valid_folder, offsets, angles, properties)

    # Initialize SVM classifier
    clf = svm.SVC()
    # Traing SVM classifier
    clf.fit(X_train, y_train)

    # Save model with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    model_path = f"models/model_{timestamp}.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_path)

    # Show all metrics for validation set using sklearn's classification report
    y_valid_pred = clf.predict(X_valid)
    print(classification_report(y_valid, y_valid_pred))
