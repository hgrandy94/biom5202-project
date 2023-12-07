## BIOM 5202 Final Project
### Classifying Knee Injuries
#### By: Phil Forster & Heather Grandy

##### Example: Running Image Processing, Model Training, and Evaluation



##### Example: Running Image Segmentation, Model Training, and Evaluation

##### Image Processing Scripts

- preprocessing_functions.py: This script contains four image preprocessing steps to be used in our image processing pipeline. When executing the image processing pipeline (data_loading.py), functions within this script are called by the pipeline function.
    
    apply_gaussian_filter: Applies a Gaussian filter to the input image. Sigma and kernel size must be  specified as input parameters.

    apply_median_filter: Applies a median filter to the input image. Kernel size must be specified as
    an input parameter.

    canny_edge: Performs Canny edge detection on the input image. Sigma, the standard deviation for the Gaussian distribution used as part of the Canny edge detection process must be specified.

    histogram_equalization: Applies histogram equalization to the input image.

- [NOT USED] segmentation_functions.py: Functions were developed for image segmentation-related tasks. This included functions for K-Means segmentation, GLCM textural descriptor extraction and analysis, Gabor filtering, and Linear Binary Patter (LBP) filtering. After no success getting these functions working effectively across multiple images, a new set of scripts were developed specifically for image segmentation, described later.

- pipeline.py: The pipeline contains a single function called pipeline which runs specified image preprocessing or image segmentation tasks, depending on what is specified within the function. This function was modified after each pipeline iteration by commenting/uncommenting various preprocessing tasks. This script also contains setup variables for GLCM + SVM segmentation - namely, the offsets, angles, properties, model, and patch size. This was done to avoid setting these values for each image slice (i.e. every time the pipeline function is called from the data_loading.py script) which would introduce additional computation time. Though this is not best practice, it was a simple fix to allow us to achieve the goals of our project.

- data_loading.py: This is the main script that orchestrates the image processing pipeline. It iterates through all the MRI exams, passing each image slice to the pipeline function for processing. The resulting images are stored in a directory with the same structure as the MRNet dataset to facilitate model training. This was developed with assistance from ChatGPT.

##### Image Segmentation Scripts & Assets

###### Scripts

- label.py: This script facilitates image labeling. It allows the user to toggle through each MRI exam (.npy file) as well as each slice. The right and left arrow keys enabling switching between MRI exams while the up and down arrows allow the user to move through each image slice. Points in an image slice can be labeled using a right click, which will appear as red dots. A single region of interest can be drawn as a box (this functionality ended up not being used in our project). The script saves all labels to a file called "annotations_date_timestamp.json," storing the .npy file paths, corresponding image slices, and x,y positions of the labels. The escape button should be used to exit the display window pop-up.

    The label.py script was used to label approximately 2000 points to facilitate texture segmentation. The script was developed with significant help from ChatGPT. 

- segmentation_data_loading.py: This function is very similar to data_loading.py with some segmentation-specific modifications. A function called process_file was introduced which ensures only sagittal images are passed to the image segmentation pipeline. After segmentation, the resulting segmentation masks are added to a numpy array using the "stack" function. This format facilitates downstream model training. Axial and coronal images are simply copied to the desired output directory.

- prepare_patches.py: This script uses the annotations generated using label.py and prepares image "patches," i.e. small chunks of images to be used for training and validating an SVM model.

    parse_annotation: This function opens an annotation file with labels in JSON format and returns them in a list format for further use.
    
    extract_patches: This function creates pulls small "patches" from the images, using the previously generated labels in list format from parse_annotation. 

- segmentation_train.py: This script computes GLCM features using the previously generated image patches and uses them to train an SVM classifier. The model is stored for future use.

    calculate_gray_level_comatrix_stats: This function computes GLCM feature statstics and returns them in an array. This function is called from within generate_features.

    generate_features: This function takes in a dataset (defined by df and folder) and produces the GLCM features and labels.

- segment.py: This script contains a single function to segment an image using the GLCM features and an SVM classifier. First, the image has to be split into chunks - as described in the project report, this was required to reduce computation time. Then the calculate_gray_level_comatrix_stats function is used to compute the GLCM textural descriptors for the image chunks. These textural descriptors are passed as input features into the trained SVM model. Using the model, the segmentation mask is obtained. 

###### Assets

- models: This folder contains the SVM model generated as part of the texture segmentation with GLCM process. It is the model used to perform the segmentation step. If the script were to be re-ran, additional models would appear in this folder.

- output_annotations: This folder contains the json files with the labels generated using label.py. annotations_20231126_195639.json contains the labels for the ACL region whereas negative1.json and negative2.json contain the labels for the non-ACL regions. 

- segmentation_data

##### Triple MRNet Scripts

- loader.py: This script loads and prepares MRI data for CNN model training. Rescaling images to be the same size, standardizing images, and scaling the min/max pixel values to be between 0 and 255. Then, the images are stacked into a multi-dimensional array to simulate a 3-channel image. In the case of segmentation, the sagittal view images use the segmentation mask as the third image channel (vs. duplicating the image array three times, as is done in all other cases). Tensors are created for each view (axial, sagittal, coronal) which are used for model training.

- evaluation.py: This script evaluates a dataset on a trained model. It requires the following information to be passed in:
    - datadir: Data directory where the images to be used for evaluation are saved.
    - model_path: Path where the model of interest is saved.
    - output_path: Path where the model evaluation results csv is saved.

- train.py: This script produces CNN models. At each epoch, the model is saved in an output folder if it is an improvement on the previous epoch in terms of the training loss metrics. To run this script, the following information must be specified in the command-line argument:
    - datadir: Data directory where the images to be used for training are saved.
    - task: Defines what task is being run - in the original Triple MRNet model, these were "abnormal", "meniscus", and "acl". For the purposes of our project, only "acl" was tested.
    - output_path: Defines where the generated models and parameters will be saved.
    - The argument "gpu" is passed at the end to ensure the script takes advantage of GPU hardware.

##### Helper Functions and Notebook
- file_restructuring.py: For some of the image processing steps, the file structure and type was modified. To later ensure the processed images could be easily inputted into Triple MRNet, this script was created to reorganized the dataset into the original directory structure.

- visualization_functions.py: Visualizing images is a computationally intensive step in Python and therefore it is undesirable to incude this functionality in scripts where visualization is not required. However, it can be very useful to generate plots, and so two functions were developed to enable easy plotting of any data. A single matplotlib plot or a matplotlib subplot with two plots can be obtained using this script.

- test.ipynb: This notebook was used to test image processing functionality before processing all images in the MRNet dataset. This can be considered our "experimentation" place.

##### REFERENCES