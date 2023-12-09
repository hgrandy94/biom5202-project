import numpy as np
import matplotlib.pyplot as plt

sets = ["train","valid"]
views = ["sagittal","coronal","axial"]
base_folder1= "MRNet-v1.0"
base_folder2= "D:\\MRNet_threshold_new_npy"
set = sets[0]

i = 1131
i2 = 1130
j=1
if(i>=1130):
    set = sets[1]

img_array = np.load(f"{base_folder1}\\{set}\\{views[0]}\\{i:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[16,:,:],cmap='gray')
plt.axis('off')
j+=1

img_array = np.load(f"{base_folder1}\\{set}\\{views[1]}\\{i:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[11,:,:],cmap='gray')
plt.axis('off')
j+=1

img_array = np.load(f"{base_folder1}\\{set}\\{views[2]}\\{i:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[15,:,:],cmap='gray')
plt.axis('off')
j+=1

img_array = np.load(f"{base_folder2}\\{set}\\{views[0]}\\{i2:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[16,:,:],cmap='gray')
plt.axis('off')
j+=1

img_array = np.load(f"{base_folder2}\\{set}\\{views[1]}\\{i2:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[11,:,:],cmap='gray')
plt.axis('off')
j+=1

img_array = np.load(f"{base_folder2}\\{set}\\{views[2]}\\{i2:04d}.npy")
plt.subplot(2,3,j)
plt.imshow(img_array[15,:,:],cmap='gray')
plt.axis('off')
j+=1



plt.show()
