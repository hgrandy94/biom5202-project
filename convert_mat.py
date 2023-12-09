import numpy as np
import glob
import os
import time
base_folder= "MRNet-V1.0"
sets = ["train","valid"]
views = ["sagittal","coronal","axial"]
num_scans_train = 1130
num_scans_valid=120
new_folder = "MRNet_threshold"
import image_pro_functions as im;

# for f in npzFiles:
#     fm = os.path.splitext(f)[0]+'.mat'
#     d = np.load(f)
#     savemat(fm, d)
#     print('generated ', fm, 'from', f)
time1=time.time()
os.mkdir(f"D:\\{new_folder}")

for set in sets:
    os.mkdir(f"D:\\{new_folder}\\{set}")
    for view in views:
        os.mkdir(f"D:\\{new_folder}\\{set}\\{view}")
        
    
        if(set ==sets[0]):
            num_scans= num_scans_train
            start=0
        else:
            num_scans= num_scans_train+ num_scans_valid
            start=num_scans_train+1
        
        for i in range(start,num_scans):
            os.dir(f"D:\\{new_folder}\\{set}\\{view}")
            
            if i%10==0:
                print(f"{i} images completed")
        print(f"{view} view complete")    

time2 = time.time()

time_elapsed = time2-time1
print(time_elapsed)