from cv2 import imwrite
import cv2 as cv
import numpy as np
# from skimage.filters import threshold_multiotsu

def save_as_png(im,sample,new_folder,set,view):
    imgsize1,imgsize2, imgsize3 = im.shape
    for j in range(imgsize1):
        imwrite(f"D:\\{new_folder}\\{set}\\{view}\\{sample:04d}-{j:02d}.png",im[j,:,:])

def canny(im,sample,new_folder,set,view):
    imgsize1,imgsize2, imgsize3 = im.shape
    new_im = im
    for j in range(imgsize1):
        new_im[j,:,:] = cv.Canny(im[j,:,:],imgsize2,imgsize3)
    np.save(f"D:\\{new_folder}\\{set}\\{view}\\{sample:04d}",new_im)

def histeq(im,sample,new_folder,set,view):
    imgsize1,imgsize2, imgsize3 = im.shape
    new_im = im
    for j in range(imgsize1):
        new_im[j,:,:] = cv.Canny(im[j,:,:],imgsize2,imgsize3)
    np.save(f"D:\\{new_folder}\\{set}\\{view}\\{sample:04d}",new_im)

# def otsu_threshold(im,sample,new_folder,set,view):
#     imgsize1,imgsize2, imgsize3 = im.shape
#     new_im = im
#     for j in range(imgsize1):
#         thresholds = threshold_multiotsu(im)
#         new_im[j,:,:] =
#     np.save(f"D:\\{new_folder}\\{set}\\{view}\\{sample:04d}",new_im)


# def imsho 