import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def mse(image_target, image):
    """ Compute Mean Squared Error (MSE) """
    return np.sqrt(np.mean((image_target - image) ** 2))

def psnr(image_target, image, data_range=1.):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    image_target = np.clip(image_target, 0, data_range)
    image = np.clip(image, 0, data_range)
    
    if len(image_target.shape) == 2:
        return compare_psnr(image_target, image, data_range=data_range)
    
    if (image_target.shape[-1] != 1) or (image_target.shape[-1] != 3):
        return psnr_mc(image_target, image, data_range)
    else:
        return compare_psnr(image_target, image, data_range=data_range)

def ssim(image_target, image, data_range=1.):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        image_target, image, multichannel=False, data_range=data_range
    )

def psnr_mc(target, image, data_range=1.):
    '''
    shape: (h,w,c)
    '''
    psnr_i = 0
    for k in range(target.shape[-1]):
        psnr_i += compare_psnr(target[...,k], image[...,k], data_range=data_range)
    return psnr_i/target.shape[-1]
        
def batch_PSNR(Iclean, Img, data_range=1.):
    Img = Img.cpu().numpy().clip(0., data_range)
    Iclean = Iclean.cpu().numpy().clip(0., data_range)
    res = 0
    for i in range(Img.shape[0]):
        res += psnr(Iclean[i, ...], Img[i, ...], data_range)
    return res/Img.shape[0]

def batch_SSIM(Iclean, Img, data_range=1.):
    Img = Img.cpu().numpy().clip(0., data_range)
    Iclean = Iclean.cpu().numpy().clip(0., data_range)
    res = 0
    for i in range(Img.shape[0]):
        res += ssim(Iclean[i,...], Img[i,...], data_range)
    return res/Img.shape[0]

def tensor2numpy(image):
    image = image.detach().cpu()
    image = image.permute(0,2,3,1).numpy()
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    return image

from collections import OrderedDict
def process_weights(weight_dict):
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return  new_state_dict

def load_model(model, checkpoint):
    checkpoint = process_weights(checkpoint)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    return model

import torch
import torch.nn.functional as F
def pad_image(image, n=64):
    b,c,h,w = image.shape
    pad_w = 0
    pad_h = 0
    if w%n != 0:
        image = F.pad(image, (0,n-w%n))
        pad_w = n-w%n
        
    if h%n != 0:
        image = F.pad(image, (0,0,0,n-h%n))
        pad_h = n-h%n
    return image, pad_h, pad_w
