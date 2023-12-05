## BIOM 5202 Final Project
### Classifying Knee Injuries
#### By: Phil Forster & Heather Grandy

##### Image Processing Scripts

- preprocessing_functions.py: This script contains four image preprocessing steps to be used in our image processing pipeline. When executing the image processing pipeline (data_loading.py), functions within this script are called by the pipeline function.
    
    apply_gaussian_filter: Applies a Gaussian filter to the input image. Sigma and kernel size must be  specified as input parameters.

    apply_median_filter: Applies a median filter to the input image. Kernel size must be specified as
    an input parameter.

    canny_edge: Performs Canny edge detection on the input image. Sigma, the standard deviation for the Gaussian distribution used as part of the Canny edge detection process must be specified.

    histogram_equalization: Applies histogram equalization to the input image.

- [NOT USED] segmentation_functions.py: Functions were developed for image segmentation-related tasks. This included functions for K-Means segmentation, GLCM textural descriptor extraction and analysis, Gabor filtering, and Linear Binary Patter (LBP) filtering. After no success getting these functions working effectively across multiple images, a new set of scripts were developed specifically for image segmentation, described later.

- pipeline.py: The pipeline contains a single function called pipeline which runs specified image preprocessing or image segmentation tasks, depending on what is specified within the function. This function was modified after each pipeline iteration by commenting/uncommenting various preprocessing tasks. This script also contains setup variables for GLCM + SVM segmentation - namely, the offsets, angles, properties, model, and patch size. This was done to avoid setting these values for each image slice (i.e. every time the pipeline function is called from the data_loading.py script) which would introduce additional computation time. Though this is not best practice, it was a simple fix to allow us to achieve the goals of our project.

- data_loading.py: This is the main script that conducts the image processing pipeline. This function was developed with assistance from ChatGPT

##### Image Segmentation Scripts & Assets

###### Scripts

- label.py

- segmentation_data_loading.py

- prepare_patches.py

- segmentation_train.py

- segment.py

###### Assets

- models

- output_annotations

- segmentation_data

##### Triple MRNet Scripts

- loader.py

- evaluation.py

- train.py


##### Helper Functions and Notebook
- file_restructuring.py

- visualization_functions.py

- test.ipynb: This notebook was used to test image processing functionality before processing all images in the MRNet dataset. This can be considered our "experimentation" place.