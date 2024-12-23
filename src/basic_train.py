import cv2
import numpy as np
import torch
import h5py
import tifffile
from scipy.io import loadmat, savemat

def to_tensor(image):
    '''
    image shape: (h, w, c)
    '''
    Img_clean = torch.tensor(image).unsqueeze(0).float()
    if len(Img_clean.shape) == 3:
        Img_clean = Img_clean.unsqueeze(1)
    else:
        Img_clean = Img_clean.permute(0,3,1,2)
    return Img_clean

def load_image(path):
    if '.tif' in path or '.tiff' in path:
        image_np = tifffile.imread(path)
        image_np = image_np.transpose(2,1,0)
    elif '.mat' in path:
        try:
            f = h5py.File(path)
            image_np = np.array(f[list(f.keys())[-1]])
            f.close()
            image_np = image_np.transpose(2,1,0)
        except:
            image_np = loadmat(path)['X']
            
    image_np = image_np.astype(np.float32)
    
    for k in range(image_np.shape[-1]):
        image_np[...,k] = image_np[...,k]/image_np[...,k].max()
    
    return image_np