import os,io
from cv2 import imwrite
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt;
import image_pro_functions as im;
import os,csv
message_sagittal=""
message_coronal=""
message_axial=""


# from skimage import data
# from skimage.filters import threshold_multiotsu



num_scans_train = 1130
num_scans_valid=120
new_split = [1000,125,125]
sets = ["train","valid"]
views = ["sagittal","coronal","axial"]
base_folder= "MRNet-V1.0"
diagnoses = ["normal","acl","meniscus"]
new_folder = "MRNet_threshold"

file = open(f"{base_folder}\\{sets[0]}-{diagnoses[1]}.csv", "r")
acl_diagnoses1 = np.array(list(csv.reader(file,delimiter=",")))[:,1]
file.close()

file = open(f"{base_folder}\\{sets[1]}-{diagnoses[1]}.csv", "r")
acl_diagnoses2= np.array(list(csv.reader(file,delimiter=",")))[:,1]

acl_diagnoses = np.concatenate((acl_diagnoses1,acl_diagnoses2))


set=sets[0]
#os.mkdir(f"D:\\{new_folder}")

for set in sets:
    #os.mkdir(f"D:\\{new_folder}\\{set}")
    for view in views:
        #os.mkdir(f"D:\\{new_folder}\\{set}\\{view}")
        if(set ==sets[0]):
            num_scans= num_scans_train
            start=0
        else:
            num_scans= num_scans_train+ num_scans_valid
            start=num_scans_train+1
        
        for i in range(start,num_scans):
            img_array = np.load(f"{base_folder}\\{set}\\{view}\\{i:04d}.npy")
            #im.canny(img_array,i,new_folder,set,view)
            imgsize1,imgsize2, imgsize3 = img_array.shape
            if(view==views[0]):
                message_coronal =f"{message_coronal}{imgsize1}\n"
            elif(view==views[2]):
                message_axial =f"{message_axial}{imgsize1}\n"
            

        if(view==views[0]):
            with open('Sagittal.txt', 'w') as f:
                f.write(message_sagittal)
        elif(view==views[1]):
            with open('Coronal.txt', 'w') as f:
                f.write(message_coronal)
        elif(view==views[2]):
            with open('Axial.txt', 'w') as f:
                f.write(message_axial)       
            
            if i%10==0:
                print(f"{i} images completed")
        print(f"{view} view complete")

print("ok")

