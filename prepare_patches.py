import os
import json
import numpy as np
import pandas as pd

# Go through an annotation file and pull out the information
def parse_annotation(annotation_file, label: 1):
    """
    This function opens an annotation file with labels in JSON format and returns them
    in a list format for further use.

    Inputs
    --------
    annotation_file : json file
        Contains points to facilitate labeling images
    label : int
        Whether the labels represent ACLs (1) or not ACLS (2)

    Output
    -------
    annotations : list
        List of annotations/labels
    """
    # Open annotation file
    with open(annotation_file) as f:
        annotation_data = json.load(f)

    # Initialize list to store annotations
    annotations = []
    # Iterate through items, storing the filepath, slice number, x, y positions 
    # (labels), and the label indicator
    for filepath, slices in annotation_data.items():
        for slice_num, slice_data in slices.items():
            points = slice_data.get("points")
            if points:
                for x, y in points:
                    annotations.append(
                        (filepath, int(slice_num), int(x), int(y), label)
                    )

    return annotations


def extract_patches(df, folder, size=16):
    """
    This function creates pulls small "patches" from the images, using the
    previously generated labels. 

    Inputs
    --------
    df : pandas DataFrame
        Contains filepath, slice, x, y positions for points and label to indicate
        whether the position represents the ACL or not.
    
    folder : string
        Name of folder to store output patches
    
    size : int
        Size of patches
    
    Outputs
    --------
    Saves image patches to specified folder, no return value.

    """
    os.makedirs(folder, exist_ok=True)

    radius = size // 2
    for index, row in df.iterrows():
        filepath = row["filepath"]
        slice = row["slice"]
        x = row["x"]
        y = row["y"]

        image = np.load(filepath)

        padded_image = np.pad(image[slice], radius, mode="reflect")
        patch = padded_image[y : y + 2 * radius, x : x + 2 * radius]

        patch_path = os.path.join(folder, f"{index:04}.npy")
        np.save(patch_path, patch)

        if index % 100 == 0:
            print(f"Extracted {index} patches")

## INITIALIZE FILEPATHS
# Set file paths for positive samples (ACLs) and negative samples (not ACLS)
positive_samples_paths = ["output_annotations/annotations_20231126_195639.json"]
negative_samples_paths = [
    "output_annotations/negative1.json",
    "output_annotations/negative2.json"]

train_folder = "segmentation_data/train"
valid_folder = "segmentation_data/valid"
csv_folder = "segmentation_data/img_patches"

# Load annotations into lists using the parse_annotation function
positive_samples = []
for positive_samples_path in positive_samples_paths:
    positive_samples += parse_annotation(positive_samples_path, label=1)

negative_samples = []
for negative_samples_path in negative_samples_paths:
    negative_samples += parse_annotation(negative_samples_path, label=0)

print(f"Positive samples: {len(positive_samples)}")
print(f"Negative samples: {len(negative_samples)}")

# Save the annotations to a dataframe to facilitate downstream processing
columns = ["filepath", "slice", "x", "y", "label"]
data = positive_samples + negative_samples
df = pd.DataFrame(data, columns=columns)

# Shuffle dataframe before splitting into train/valid 
df = df.sample(frac=1).reset_index(drop=True)

# Split into train and validation sets - 80/20 train/valid split
train_df = df.iloc[: int(len(df) * 0.8)].reset_index(drop=True)
valid_df = df.iloc[int(len(df) * 0.8) :].reset_index(drop=True)
print(f"Train samples: {len(train_df)}")
print(f"Valid samples: {len(valid_df)}")

# Extract the patches and store them in the train and valid folders
extract_patches(train_df, train_folder)
extract_patches(valid_df, valid_folder)

# Save dataframes
train_df.to_csv(os.path.join(csv_folder, "train.csv"), index=False)
valid_df.to_csv(os.path.join(csv_folder, "valid.csv"), index=False)
