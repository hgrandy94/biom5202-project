import numpy as np
from sklearn import cluster
from skimage import morphological_chan_vese

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
    kmeans_cluster = cluster.KMeans(n_clusters=k)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # Scale result to the range 0-255
    newimage = cluster_centers[cluster_labels].reshape(h, w)*255.0
    newimage = newimage.astype('uint8')

    return newimage