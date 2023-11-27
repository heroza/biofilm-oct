from fun import *
import matplotlib.pyplot as plt
from skimage import morphology
from OctCorrection import *
from ImageProcessing import *
import os
import torch
import pandas as pd

# Load images from folder
folder_path = '/home/rh22708/darpa/dataset/pat/plate 9/216/'
df1, original_images = load_images(folder_path)

# Display the shape of the image array
print("\nImage Array Shape:", original_images.shape)

# correcting OCT
# blur
blur = np.array([cv2.GaussianBlur(image, (0,0), sigmaX=33, sigmaY=33) for image in original_images])
# divide
processed_images = np.array([cv2.divide(image, blur[idx], scale=255) for idx, image in enumerate(original_images)])
# otsu threshold
processed_images = np.array([cv2.threshold(image[0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for image in processed_images.astype("uint8")])
# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
processed_images = np.array([cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) for image in processed_images])

# # refer https://gitlab.com/FlumeAutomation/automated-oct-scans-acquisition
# crop=0  #find here the best position to crop the image and avoid spurious signals such as the Zspacer
# bckHeight=50 #find here the position of the line above which the background intensity is calculated
# zHeight=1024
# modelHeight=1024
# processed_images = np.array([Correct(image.swapaxes(2,1), bckHeight=bckHeight, crop=int(crop),size=image.swapaxes(2,1).shape, retFiltSize=int(12/4), GaussFiltSize=10, zHeight=zHeight,modelHeight=modelHeight)[0] for image in original_images])

# # segment using SAM
# # pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# # pretrained weight wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# CHECKPOINT_PATH = os.path.basename("/home/rh22708/darpa/sam_vit_h_4b8939.pth")
# img_rgb_array = np.array([cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in original_images[:,0]]).astype(np.uint8)
# input_point = np.array([[90, 100],[100, 900]]) # prompt points (X,Z)
# input_label = np.array([0,0]) #corresponding label: 0 negative label, 1 positive label
# processed_images = sam_predict(CHECKPOINT_PATH, img_rgb_array, input_point, input_label, multimask_output = False) # use sampe prompt for all images

# show comparable images
plt.subplot(2, 2, 1)
plt.imshow(original_images[0][0], cmap='gray')
plt.title('Original Image 1')
plt.subplot(2, 2, 2)
plt.imshow(processed_images[0], cmap='gray')
plt.title('Processed Image 1')
plt.subplot(2, 2, 3)
plt.imshow(original_images[-1][0], cmap='gray')
plt.title('Original Image -1')
plt.subplot(2, 2, 4)
plt.imshow(processed_images[-1], cmap='gray')
plt.title('Processed Image -1')
plt.tight_layout()
plt.show()

# calculating measures
# thr=0     #set here the gray value threshold to calculate the binary mask
# small_obj=30   #set here the minimum size of objects to be kept in the binary mask
# cnames=['volume','thickness','Rq', 'Ra', 'density dist.', 'pixel-based density']
# df2 = pd.DataFrame(columns =cnames )
# for image in processed_images:
#     V,H,Rq,Ra, SL, density=CalcZMap(image[None,:],thr,small_obj) 
#     df2=pd.concat([df2, pd.DataFrame([[V,H,Rq,Ra, SL, density]], columns = cnames)])   #append volume, height and roughness results in a dataframe
# df2.reset_index(inplace=True, drop=True)
# df = pd.concat([df1, df2], axis=1)
# print(df)
# print(df2)