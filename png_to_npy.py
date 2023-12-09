import numpy as np
import glob
import os
import time
import cv2 as cv
base_folder = "MRNet_threshold_new"
sets = ["valid"]
views = ["sagittal","coronal","axial"]
num_scans_train = 1130
num_scans_valid=120

new_folder = "MRNet_threshold_new_npy"
import image_pro_functions as im;

with open("Sagittal.txt") as file:
    sag_lines = [line.rstrip() for line in file]

with open("Coronal.txt") as file:
    cor_lines = [line.rstrip() for line in file]

with open("Axial.txt") as file:
    ax_lines = [line.rstrip() for line in file]
# for f in npzFiles:
#     fm = os.path.splitext(f)[0]+'.mat'
#     d = np.load(f)
#     savemat(fm, d)
#     print('generated ', fm, 'from', f)
time1=time.time()
#os.mkdir(f"D:\\{new_folder}")

for set in sets:
    os.mkdir(f"D:\\{new_folder}\\{set}")
    for view in views:
        os.mkdir(f"D:\\{new_folder}\\{set}\\{view}")
       
        num_scans= num_scans_train+ num_scans_valid
        start=num_scans_train
    

        
        for i in range(start,num_scans-1):
           
            if(view==views[0]):
                num_layers = int(sag_lines[i])
            elif(view==views[1]):
                num_layers =int(cor_lines[i])
            elif(view==views[2]):
                num_layers =int(ax_lines[i])
           
            img_array= np.empty((num_layers,256,256))
            for j in range(num_layers):
                img_array[j,:,:] = cv.imread(f"D:\\{base_folder}\\{set}\\{view}\\{i+1:04d}-{j:02d}.png",0)
        
            np.save(f"D:\\{new_folder}\\{set}\\{view}\\{i:04d}",img_array)
            if i%10==0:
                print(f"{i} images completed")
        print(f"{view} view complete")    

time2 = time.time()

time_elapsed = time2-time1
print(time_elapsed)