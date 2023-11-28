import numpy as np
from sklearn import cluster
from skimage import img_as_ubyte, filters, feature
from skimage.feature import greycomatrix
from skimage.segmentation import morphological_chan_vese

def generate_circular_seed_masks(img, center, radius):
    """
    This function returns a soft mask and binary mask in the shape of a circle to facilitate image segmentation
    using the active contour approach.

    Inputs
    ------------
    img : numpy array
        The image for which you wish to create the mask
    center : list or tuple
        The x, y coordinates for the desired centre of the circle
    radius : int
        The desired radius of the mask

    Outputs
    -------------
    soft_mask:

    binary_mask:
    """
    # Create a meshgrid of the same shape as the image
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    # Calculate the distance of each point from the center of the circle
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Set the initial level set to be the negative of this distance
    soft_mask = -1 * (dist - radius)

    # Threshold the soft mask at 0 to get a binary mask
    binary_mask = soft_mask > 0

    return soft_mask, binary_mask


def mcv(image, binary_mask, num_iter=100, smoothing=1):
    """
    This function performs image segmentation using the morphological
    Chan-Vese algorithm
    """
    mcv = morphological_chan_vese(
        image,
        num_iter=num_iter, # Can increase the number of iterations to potentially improve results
        smoothing=smoothing,
        lambda1=1,  # outer region weight
        lambda2=1,  # inner region weight
        init_level_set=binary_mask,
    )
    return mcv

def kmeans_segmentation(img, k=5):
    # Get the height and width of the image
    h, w = img.shape

    # reshape to 1D array
    image_2d = img.reshape(h*w,1)

    # Perform k-means clustering using sklearn library
    kmeans_cluster = cluster.KMeans(n_clusters=k, n_init=10)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # Scale result to the range 0-255
    newimage = cluster_centers[cluster_labels].reshape(h, w)*255.0
    newimage = newimage.astype('uint8')

    return newimage

# BELOW functions created with help from ChatGPT
def extract_glcm_features(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    img = img_as_ubyte(img) # Convert to 8-bit for GLCM
    glcm = feature.graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
    return glcm

def analyze_glcm_features(glcm):
    # Extract relevant texture features from GLCM
    contrast = np.mean(feature.graycoprops(glcm, "contrast"))
    energy = np.mean(feature.graycoprops(glcm, "energy"))
    entropy = -np.sum(glcm * np.log2(glcm + (glcm ==0)))
    homogeneity = np.mean(feature.graycoprops(glcm, "homogeneity"))
    return [contrast, energy, entropy, homogeneity]

def gabor_filter(img, frequency=0.1, theta=0, sigma=1):
    img = img_as_ubyte(img)
    gabor_filter = filters.gabor(img, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
    return gabor_filter

def apply_lbp(img, radius=1, n_points=8):
    lbp = feature.local_binary_pattern(img, P=n_points, R=radius, method='uniform')
    return lbp