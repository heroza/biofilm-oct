import os
import pandas as pd
import numpy as np
from fun import *
from OctCorrection import *
from ImageProcessing import *
import sys

def load_csv_images(folder_path):
    image_array = []
    file_paths = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, filename)
            file_paths.append(os.path.splitext(filename)[0])

            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(file_path, header=None)

            # Convert DataFrame to NumPy array
            image_data = df.to_numpy()

            # Append to the image array
            image_array.append(image_data)

    df_paths = pd.DataFrame({'FilePath': file_paths})
    return np.array(image_array), df_paths

plate = sys.argv[1]
folder_path = "/home/rh22708/darpa/dataset/annotated/Plate "+plate+'/'
annotated_images, df1 = load_csv_images(folder_path)

# Display the shape of the resulting array
print("Shape of the loaded annotated images array:", annotated_images.shape)

thr=0     #set here the gray value threshold to calculate the binary mask
small_obj=30   #set here the minimum size of objects to be kept in the binary mask
cnames=['plate','volume','thickness','Rq', 'Ra', 'density dist.', 'pixel-based density']
df2 = pd.DataFrame(columns =cnames )
for idx, image in enumerate(annotated_images):
    fig, V,H,Rq,Ra, SL, density=CalcZMap(image[None,:],thr,small_obj) 
    df2=pd.concat([df2, pd.DataFrame([[plate,V,H,Rq,Ra, SL, density]], columns = cnames)])   #append volume, height and roughness results in a dataframe
    plt.savefig(folder_path+df1.loc[idx, 'FilePath']+"_output"+'.png')
df2.reset_index(inplace=True, drop=True)
df = pd.concat([df1, df2], axis=1)
# Save the DataFrame to a CSV file
df.to_csv(plate+'.csv', index=False)