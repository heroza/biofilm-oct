from fun import *
import matplotlib.pyplot as plt
from skimage import morphology
from OctCorrection import *
from ImageProcessing import *
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Load images from folder
folder_path = '/home/rh22708/darpa/dataset/pat/plate 9/216/'
df, image_array = load_images(folder_path)

# image_array = np.load('output_file.npy')
# df = pd.read_csv('output_dataframe.csv')

# Display the dataframe
# print("DataFrame:")
# print(df)

# Display the shape of the image array
print("\nImage Array Shape:", image_array.shape)

# correcting OCT
# refer https://gitlab.com/FlumeAutomation/automated-oct-scans-acquisition
# crop=10  #find here the best position to crop the image and avoid spurious signals such as the Zspacer
# bckHeight=50 #find here the position of the line above which the background intensity is calculated
# zHeight=1014
# modelHeight=1000
# image_array = np.array([Correct(image.swapaxes(2,1), bckHeight=bckHeight, crop=int(crop),size=image.swapaxes(2,1).shape, retFiltSize=int(12/4), GaussFiltSize=10, zHeight=zHeight,modelHeight=modelHeight)[0] for image in image_array])

# segment using SAM
# pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# pretrained weight wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
CHECKPOINT_PATH = os.path.basename("/home/rh22708/darpa/sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)
img_rgb = cv2.cvtColor(image_array[0,0], cv2.COLOR_GRAY2RGB).astype(np.uint8) # first image first layer. as far as I knwo, SAM only get RGB image
mask_predictor.set_image(img_rgb)
input_point = np.array([[90, 230],[100, 330]]) # prompt points
input_label = np.array([1,0]) #corresponding label: 0 negative label, 1 positive label
masks, scores, logits = mask_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False, # False for include only the best mask
)
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

# calculating measures
# thr=25     #set here the gray value threshold to calculate the binary mask
# small_obj=30   #set here the minimum size of objects to be kept in the binary mask
# cnames=['fname', 'volume','height','roughness']
# df = pd.DataFrame(columns =cnames )
# for image in image_array:
#     mask,Z,V,H,R=CalcZMap(image,thr,small_obj) 
#     df=df.append(pd.DataFrame([[path, V,H,R]], columns = cnames))   #append volume, height and roughness results in a dataframe
# print(df)