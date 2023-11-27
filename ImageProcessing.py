import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
from PIL import Image
import cv2
from numba import jit

def f1(a):
    try:
        x=a.shape-np.where(a)[0][0]
    except:
        x=0
    return x
    
def f2(a):
    try:
        x=a.shape-np.where(a)[0][-1]
    except:
        x=0
    return x
        
def _CalcZMap(image, thr, small_obj):
    mask=morphology.remove_small_objects(image>thr,small_obj)    #exclude objects smaller than 30 pixels, can be customized
    Z=np.squeeze(np.apply_along_axis(f1,1, mask))
    #z2=np.squeeze(np.apply_along_axis(f2,1, mask))
    V=np.sum(mask)    #calc volume as sum of 1 in mask, excluding plexi
    H=np.mean(Z[Z>0])
    R=np.std(Z[Z>0])/H
    return mask,Z,V,H,R#,z2

def CalcZMap(image, thr, small_obj): # mask is 2d
    mask=morphology.remove_small_objects(image>thr,small_obj)
    mask = mask[0]
    surface = mask.argmax(axis=0)
    substrate = mask.shape[0] - np.flip(mask, axis=0).argmax(axis=0) - 1
    fig,ax = plt.subplots(1)
    im = ax.imshow(mask,cmap='gray')
    plt.plot(np.arange(surface.shape[0]), surface, ls='-', c='red', lw=1)
    plt.axhline(y=np.min(substrate), color='b', linestyle='--')
    

    ref_surf = np.min(substrate)
    mask = mask[:ref_surf]
    # recalculate surface and substrate
    surface = mask.argmax(axis=0)
    substrate = mask.shape[0] - np.flip(mask, axis=0).argmax(axis=0) - 1
    
    Z=np.squeeze(np.apply_along_axis(f1, 0, mask))
    V=np.sum(mask)    #calc volume as sum of 1 in mask
    H=np.mean(Z[Z>0])
    plt.axhline(y=ref_surf - H, color='y', linestyle='--')
    Rq = np.std(Z[Z>0])/H #Root-mean-square of heights
    Ra=np.mean(abs(Z[Z>0]-H))/H #Arithmetic average of heights, normalized.
    SL = np.sum(mask[0:int(mask.shape[0]-H/2),:]) / V
    bio_vol = np.sum(substrate-surface+1)
    density = V/bio_vol
    return fig, V,H,Rq,Ra, SL, density

def find_ol(image1,image2,ol):
    sy,sx=image1.shape
    a,b=int(sy*0.05), int(sx*0.05)
    img1=image1[:,int(sx*(1-ol)):].astype('uint8')
    temp=image2[a:-a,b:int(ol*sx)-b].astype('uint8')
    res = cv2.matchTemplate(img1,temp,eval('cv2.TM_CCOEFF_NORMED'))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h=(max_loc[1]-a,max_loc[0]-b+int(sx*(1-ol)))
    return h

def ol_mat(M, ol):
    n_rows,n_cols, sy,sx=M.shape
    OL=np.full((2,n_rows,n_cols),0)
    for row in range(n_rows):
        for col in range(n_cols-1):
            OL[0,row,col+1]=int(find_ol(M[row,col,:,:],M[row,col+1,:,:],ol)[0])
            OL[1,row,col+1]=int(find_ol(M[row,col,:,:],M[row,col+1,:,:],ol)[1])
   
    OLn=np.zeros(OL.shape)
    for row in range(n_rows):
        OLn[0,row,:]=np.asarray([int(sum(OL[0,row,:i+1])) for i in range(n_rows)])
        OLn[1,row,:]=np.asarray([int(sum(OL[1,row,:i+1])) for i in range(n_cols)])
    return OLn

def stitch(M,OL,OLv,ol,crop):
    n_rows,n_cols, sy,sx=M.shape
    #Z=np.zeros((int(sy*n_rows*(1-ol)),int(sx*n_cols*(1-ol))))
    Z=np.zeros((int(sy*n_rows),int(sx*n_cols)))
    
    for row in range(n_rows):
        for col in range(n_cols):
            ol_Y=OLv[1,col,row]+crop
            ol_x=OL[1,row,col]+crop
            im=M[row,col,crop:-crop,crop:-crop]
            Z[int(ol_Y):int(ol_Y+im.shape[0]),int(ol_x):int(ol_x+im.shape[1])]=im
    Z=Z[:int(min(OLv[1,:,n_rows-1])+crop+im.shape[0]),:int(min(OL[1,:,n_cols-1])+crop+im.shape[1])]
    return Z


