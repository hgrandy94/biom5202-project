import os,io
from cv2 import imwrite
import numpy as np
from matplotlib import pyplot as plt;
import os,csv
import splitfolders

num_scans_train = 1130
num_scans_valid=120
new_split = [1000,125,125]
sets = ["train","valid"]
views = ["axial","coronal","sagittal"]
base_folder= "MRNet-V1.0"
new_folder_acl = "Knee MRIs restructured_acl"
diagnoses = ["normal","acl","meniscus"]
file = open(f"{base_folder}\\{sets[0]}-{diagnoses[1]}.csv", "r")
acl_diagnoses1 = np.array(list(csv.reader(file,delimiter=",")))[:,1]
file.close()

file = open(f"{base_folder}\\{sets[1]}-{diagnoses[1]}.csv", "r")
acl_diagnoses2= np.array(list(csv.reader(file,delimiter=",")))[:,1]

acl_diagnoses = np.concatenate((acl_diagnoses1,acl_diagnoses2))
acl_diagnoses.astype(int)
print(acl_diagnoses)

os.mkdir(f"{new_folder_acl}")
os.mkdir(f"{new_folder_acl}\\normal")
os.mkdir(f"{new_folder_acl}\\torn")

set=sets[0]

for view in views:
    set=sets[0]
    for i in range(num_scans_valid+num_scans_train):
            if(i==num_scans_train):
                set=sets[1]
            img_array = np.load(f"{base_folder}\\{set}\\{view}\\{i:04d}.npy")
            if(acl_diagnoses[i]=='1'):
                diagnosis='torn'
            else:
                diagnosis='normal'
            imwrite(f"{new_folder_acl}\\{diagnosis}\\{view}{i:04d}.png",img_array[10,:,:])
            if i%100==0:
                print(f"{i} images completed")

splitfolders.ratio(new_folder_acl,output=f"{new_folder_acl}-Split", ratio=(0.8, 0.1, 0.1))

print("ok")
