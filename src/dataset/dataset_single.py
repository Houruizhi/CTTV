import os
import cv2
import math
import torch
import random
import numpy as np
from tqdm import tqdm

def random_crop_image(img, patch_size=64):
    h, w = img.shape[-2:]

    hi = np.random.randint(0, h - patch_size + 1) 
    wi = np.random.randint(0, w - patch_size + 1)

    cropped = crop_image(img, (hi, hi + patch_size), (wi, wi + patch_size))
    return cropped

def crop_image(img, crop_h=(0,0), crop_w=(0,0)): 
    '''
    input img size: (n,c,h,w)
    Args:
        crop_h: the begin and end location along the height axis.
        crop_w: the begin and end location along the width axis.
    '''
    assert 2 <= len(img.shape)
    return img[...,crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]