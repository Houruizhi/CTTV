import numpy as np
import os
import cv2
import time

import torch
import torch.nn as nn
from torch.optim import Adam

import torch
import torch.nn.functional as F

from src.basic_train import load_image, savemat, to_tensor
from src.utils import psnr, tensor2numpy

def add_noise_HSI(Img_clean, gaussian=True, sp=True, stripe=True, deadline=True):
    '''
    image: N,Band,Height,Width
    '''
    torch.manual_seed(3)
    N,B,H,W = Img_clean.shape
    Img_noisy = Img_clean.clone()
    if gaussian:
        noise_level = 0.1 + torch.rand(1,B,1,1).to(Img_noisy.device)*0.2
        Img_noisy = Img_noisy + noise_level*torch.randn_like(Img_noisy)
    
    if sp:
        mask_sp = (torch.rand_like(Img_clean) < 0.2).byte()
        mask_0 = mask_sp*(torch.rand_like(Img_clean) < 0.5).byte()
        mask_1 = mask_sp - mask_0
        Img_noisy = (1-mask_sp)*Img_clean + mask_1.float()
        # Img_noisy = mask*Img_clean
    else:
        Img_noisy = Img_clean.clone()
        mask_sp = torch.zeros_like(Img_noisy)
        
    if stripe:
        # % add stripe
        all_band = torch.randperm(B)
        stripe_bands = all_band[:int(0.3*B)]
        for i in range(stripe_bands.shape[0]):
            loc = (torch.rand(1,1,W) < 0.2).byte().to(Img_noisy.device)
            noise = torch.rand(1,1,W).to(Img_noisy.device)*0.5-0.25
            noise = loc*noise
            Img_noisy[:,stripe_bands[i],...] = Img_noisy[:,stripe_bands[i],...] - noise
            mask_sp[:,stripe_bands[i],:,:] = mask_sp[:,stripe_bands[i],:,:]*(1-loc) + loc
    else:
        stripe_bands = torch.randperm(B)
    
    if deadline:
        all_band = torch.randperm(B)
        deadline_bands = all_band[:int(0.3*B)]
        for i in range(deadline_bands.shape[0]):
            loc = (torch.rand(1,1,W) < 0.1).byte().to(Img_noisy.device)
            Img_noisy[:,deadline_bands[i],...] = Img_noisy[:,deadline_bands[i],...]*(1-loc)
            mask_sp[:,deadline_bands[i],:,:] = mask_sp[:,deadline_bands[i],:,:]*(1-loc) + loc
    else:
        deadline_bands = torch.randperm(B)
        
    return Img_noisy, mask_sp.float(), stripe_bands, deadline_bands

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from src.models.dncnn_dropout import DnCNN

    cases_list = [
        [True, True, False, False],
        [True, True, True, False],
        [True, True, False, True],
        [True, True, True, True],
    ]
    data_root = '/home/rzhou/ssd_cache/HSIDataset'
    save_root = '/home/rzhou/HSIDenoising/HSIResults'
    # if os.path.exists(save_root):
    #     save_root = save_root + str(time.time())

    # datasets = ['simulate', 'CAVE', 'ICVL_val']
    # datasets = ['CAVE', 'ICVL_val']
    datasets = ['simulate']
    for dataset_name in datasets:
        files = os.listdir(os.path.join(data_root, dataset_name))
        for file_i in files:
            clean_image = load_image(os.path.join(data_root, dataset_name, file_i))
            Img_clean = to_tensor(clean_image)
            for case_k, case in enumerate(cases_list, 1):
                Img_noisy, mask_sp, stripe_bands, deadline_bands = add_noise_HSI(Img_clean, *case)
                save_dir = os.path.join(save_root, dataset_name, f'case_{case_k}', 'Noisy')
                os.makedirs(save_dir, exist_ok=True)
                print(save_dir)
                noisy = tensor2numpy(Img_noisy).squeeze()
                print(f'file: {file_i}, case: {case_k}, PSNR: {psnr(noisy, clean_image)}')
                savemat(
                    os.path.join(save_dir, file_i),
                    {
                        'noisy': noisy.squeeze(),
                        'mask_sp': tensor2numpy(mask_sp).squeeze(),
                        'stripe_bands': stripe_bands.squeeze().cpu().numpy(),
                        'deadline_bands': deadline_bands.squeeze().cpu().numpy()
                        }
                    )