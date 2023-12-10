## BIOM 5202 Final Project
### Classifying Knee Injuries
#### By: Phil Forster & Heather Grandy

This GitHub repository contains the code and additional assets generated to meet requirements for BIOM 5202 at Carleton University in the Fall 2023 semester. Our project aimed to answer the question: Do image processing techniques improve ACL injury classification? We used the Stanford University MRNet dataset and Yash Bhalgat's Triple MRNet solution for our project: https://github.com/yashbhalgat/MRNet-Competition/tree/master.

#### Step-by-Step Examples

##### Example: Running Image Processing, Model Training, and Evaluation

The two steps below were run locally on a computer with only CPU resources.

1. Update the pipeline function in pipeline.py to include only the desired image processing steps. For example, if you wished to run a pipeline iteration to test the performance of a median filter with kernel=3, uncomment the line:
img = apply_median_filter(img, kernel_size=3)
Multiple image processing steps can be added within this function, if desired. This script imports functions from the preprocessing_functions.py script.

2. Execute the data_loading.py script. This will ensure every image slice runs through the image processing pipeline and is stored in an output directory with the exact same file structure as the original MRNet dataset. Note that this script also returns the total image processing time (one of our evaluation metrics).

3. Manually copy the output directory with the processed images to the GPU server. Both were Windows machines so the files were simply copy and pasted using the file explorer.

The following steps were executed on a GPU server.

4. Run the Triple MRNet train.py script. The data directory where the newly processed images are saved must be passed as input. An output folder to store the models and model parameters must also be specified. The best model will appear as the first item in the output folder, and the performance metrics are printed to the output at the end of each epoch.

Example command:

```
python .\src\train.py --datadir "C:\MRNet-v1.0" --rundir "output_file_path" --task "acl" --gpu
```

*Note that the datadir and rundir parameters were modified as required for each pipeline iteration. Specifically, the datadir indicated where the processed images were saved and the rundir was updated for each iteration so as to not overwrite model files. The GPU argument at the end of the command ensures the train.py script knows to leverage the available GPU compute.

5. Identify the best epoch and the resulting model, and run the Triple MRNet evaluate.py script. The data directory where the processed images are saved and the path to the model must be passed as input. The train.py script saved the best performing model in terms of AUC as the first item in the output directory (rundir above) so it was straightforward to determine. The results were also printed to the terminal output and could be verified by reviewing the outputs after each epoch.

Example command:

```
python .\src\evaluate.py --datadir "input_data_directory" --model_path "path_to_best_model" --split "valid" --diagnosis acl --gpu
```

*Note that the datadir and model_path parameters were modified as required for each pipeline iteration. The datadir was updated to specify the location of the processed images (same directory as in step 4) and the model_path was updated to be the best performing model in terms of AUC, as determined in step 4. The split argument ensures the validation data is used for evaluation. The diagnosis parameter allowed us to focus only on the ACL classification (as the original Triple MRNet also supports abnormal and meniscus injury classifications). The GPU argument at the end of the command ensures the train.py script knows to leverage the available GPU compute.

In addition to printing the evaluation results to the terminal output, this script saved the results to a csv. Evaluation results csvs were combined into a single csv to facilitate comparison between each image processing pipeline run. 

##### Example: Running Textural Image Segmentation, Model Training, and Evaluation

*IMPORTANT NOTE* The steps below were only applied to sagittal images since the ACL region cannot be as easily identified in the coronal view, and is not visible in the axial view.

1. Create positive and negative labels for the images using the label.py script, described below. That is, manually label a random set of knee MRI slices to create positive labels, i.e. use the label.py script to select points that represent the ACL region (positive labels) and storing these values in a JSON file. Then, use the label.py script again to select points that do NOT represent the ACL region (negative labels) and store these values in another JSON file.

2. Run the prepare_patches.py script. In this script, the labels are first combined into a list. The labeled points list is then used to extract the associated "patches" from the images so that this information can be passed into a Support Vector Machine (SVM) classifier. The image patches and the associated labels for each (ACL region / non-ACL region) are saved to an output folder for use in the next step.

3. Run the segmentation_train.py script. Here, Gray-Level Co-occurrence Matrix (GLCM) textural descriptors are computed using skimage functionality. The GLCM features are extracted from the image patches generated in step 2. At this point, we now have textural descriptors for image patches containing and ACL region as well as image patches not containing an ACL region. Thus, we have constructed a dataset with the textural descriptor values and the associated label, which can be passed into an SVM classifier for model creation. The SVM classifier is trained and validated using sklearn capabilities. The output of this step is an SVM model that can finally be used to perform image segmentation.

4. Ensure the segment.py script is updated to load the preferred SVM model. This script accepts image slices, breaks them into chunks of size 16 (described in more detail in our report, but this was to improve computation time), and calculates the GLCM features. Once these features are obtained, they are passed to the SVM model trained in step 3, and the resulting image segmentation mask is obtained.

5. Update the pipeline.py script to call the segment_image function from the segment.py script. This will ensure each sagittal image slice is segmented. Following a Triple MRNet-like approach, all image slices went through the pipeline, even if the ACL was not visible as we did not have a method to automatically identify the "optimal" image slices. We theorized that this approach was acceptable, as the segmentation mask should show nothing for non-useful image slices, and the CNN would be able to understand this information.

6. Run the segmentation_data_loading.py script which is a modified version of the data_loading.py script. The key difference being that only sagittal images are passed to the pipeline function, while the axial and coronal images are simply copied to the output folder. Additionally, the structure of the sagittal images was modified such that the numpy array contained three channels - two duplicating the sagittal images, and the third being the segmentation masks.

7. Follow steps 3-6 from the image processing example above. Note that the Triple MRNet load.py script was modified for the segmentation pipeline to ensure the sagittal image slices are inputted as the three channel numpy array with the third channel being the segmentation mask, as described in step 6.

#### Description of Contents in this Repository
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

- file_restructuring.py: This script organizes the files in the form they need to be in to be used for training and applies an image processing technique from image_pro_functions.py if desired.

- image_pro_functions.py: This script includes a number of of image processing techniques to be used by file_restructuring

- png_to_npy.py: This script takes a set of MRI images and combines all slices from one MRI into a single npy array. This is done because the multi level otsu based segmentation needed to be performed in matlab and matlab could not work with/npy images.


###### Assets

- models: This folder contains the SVM model generated as part of the texture segmentation with GLCM process. It is the model used to perform the segmentation step. If the script were to be re-ran, additional models would appear in this folder.

- output_annotations: This folder contains the json files with the labels generated using label.py. annotations_20231126_195639.json contains the labels for the ACL region whereas negative1.json and negative2.json contain the labels for the non-ACL regions. 

- segmentation_data: The image patch information resulting from the prepare_patches.py script.

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
[1] OpenAI, "ChatGPT 3.5," [Online]. Available: https://chat.openai.com/.

[2] Y. S. Bhalgat, "MRNet-Competition," GitHub, Inc., 13 06 2019. [Online]. Available: https://github.com/yashbhalgat/MRNet-Competition.

[3] NumPy Developers, "numpy.lib.format," [Online]. Available: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html. [Accessed 15 10 2023].

[4] 	scikit-image team, "GLCM Texture Features," [Online]. Available: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html. [Accessed 30 11 2023].

[5] 	scikit-learn developers, "1.4 Support Vector Machines," [Online]. Available: https://scikit-learn.org/stable/modules/svm.html. [Accessed 30 11 2023].
