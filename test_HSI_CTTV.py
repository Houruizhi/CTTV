import numpy as np
import os
import cv2
import time

import torch
import torch.nn as nn
from torch.optim import Adam

import torch
import torch.nn.functional as F

from src.basic_train import (
    loadmat,
    load_image, 
    savemat, 
    to_tensor
)
from src.utils import psnr, ssim, tensor2numpy
from src.utils_n2s_dropout import trainer

def SAM3D(TensorT, TensorH):
    def SAM(a1,a2):
        SigmaTR = np.sum(a1*a2, axis=-1)
        SigmaT2 = np.sum(a1*a1, axis=-1)
        SigmaR2 = np.sum(a2*a2, axis=-1)
        return np.arccos(SigmaTR/(1e-3 + np.sqrt(SigmaT2*SigmaR2)))

    assert len(TensorT.shape) == 3
    assert len(TensorH.shape) == 3
    h,w,c = TensorH.shape

    sam = 0

    for i in range(h):
        T = np.squeeze(TensorT[i,...])
        H = np.squeeze(TensorH[i,...])
        sam += np.sum(SAM(T,H))
    
    return sam / (h*w)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from generate_simulate_data import add_noise_HSI
cases_list = [
    (1, [True, True, False, False]),
    # (2, [True, True, True, False]),
    # (3, [True, True, False, True]),
    # (4, [True, True, True, True]),
]
data_root = './'
save_root = './results/'

# from src.utils_n2s_NCTV_s2s import NCTV as NCTV_CNN
from src.utils_CTTV import CTTV

dataset_list = [
    'data'
]

for dataset in dataset_list:
    data_dir = os.path.join(data_root, dataset)
    files = os.listdir(data_dir)
    tau1 = 0.0012
    for file_i in files:
        clean_image = load_image(os.path.join(data_dir, file_i))
        for case_k, case in cases_list:
            Img_noisy = to_tensor(clean_image)
            Img_noisy,_,_,_ = add_noise_HSI(Img_noisy, gaussian=case[0], sp=case[1], stripe=case[2], deadline=case[3])
            # noisy_data = loadmat(os.path.join(save_root, dataset, f'case_{case_k}', 'Noisy', file_i))
            # Img_noisy = to_tensor(noisy_data['noisy'])
            save_dir = os.path.join(save_root, dataset, f'case_{case_k}', 'CTTV')
            os.makedirs(save_dir, exist_ok=True)
            
            Img_noisy = Img_noisy.clamp(0,1)
            # X,U,V,_ = NCTV_CNN(Img_noisy.permute(0,2,3,1).squeeze(0), X_ori=clean_image, tau1=tau1)
            X,U,V,PSNRs = CTTV(Img_noisy.permute(0,2,3,1).squeeze(0), X_ori=clean_image, tau1=tau1, eta=1.1, MAX_ITER=150)
            
            print(f'file: {file_i}, case: {case_k}, PSNR: {PSNR}, SSIM: {SSIM}, SAM: {SAM}')
            savemat(os.path.join(save_dir, file_i), {'denoised': X.cpu().numpy(), 'U': U.cpu().numpy()})