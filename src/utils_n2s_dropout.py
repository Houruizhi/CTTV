import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import cv2
import math
import time
import numpy as np
from tqdm import tqdm
from .utils import psnr, tensor2numpy
from .tv_op import nabla_x, nabla_y

def inference_bayesian(Img_noisy, model, confidence_map=None, alpha=0.9, N=30):
    model.eval()
    denoised_all = torch.zeros_like(Img_noisy)
    for _ in range(N):
        with torch.no_grad():
            for i in range(len(model)):
                mask = (torch.rand_like(Img_noisy) < alpha).float()
                if confidence_map is not None:
                    mask = confidence_map*mask
                denoised_all += model[i](Img_noisy, mask)
    return denoised_all / (N*len(model))

def reset_model(C,model_channel=[64],model_layers=20):
    from src.models.dncnn_dropout import DnCNN
    model = nn.ModuleList()
    for i in range(len(model_channel)):
        model.append(DnCNN(C,C,model_layers,model_channel[i],kernel_size=3,norm_type='bn').cuda())
    model.train()
    
    optimizer = Adam([{'params': model.parameters()}], lr=1e-3)
    return model, optimizer

def random_crop(images):
    n,c,h,w = images[0].shape
    h_i = np.random.randint(h//2 - 1)
    w_i = np.random.randint(w//2 - 1)
    return [image[:,:,h_i:h_i+h//2,w_i:w_i+w//2] for image in images]

def random_flip(images):
    if torch.rand(1)<0.5:
        images = [image.flip(dims=[-1]).contiguous() for image in images]
    if torch.rand(1)<0.5:
        images = [image.flip(dims=[-2]).contiguous() for image in images]
    if torch.rand(1)<0.5:
        images = [image.permute(0,1,3,2) for image in images]
    return images

def random_stripe(images):
    if torch.rand(1)<0.5:
        if torch.rand(1)<0.5:
            images = [image[...,::2] for image in images]
        else:
            images = [image[...,::2,:] for image in images]
    return images

def augment_image(*images):
    return random_flip(images)

def image_filters(bin_img, ksize=3): # 已测试 
    #先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    # 取每个 patch 中最小的值
    eroded1, _ = patches.reshape(B, C, H, W, -1).median(dim=-1)
    eroded2 = patches.reshape(B, C, H, W, -1).mean(dim=-1)
    return eroded1, eroded2

def trainer(
        Img_noisy,
        clean_image=None,
        num_iter=5000,
        print_epoch=30,
        data_range=1.,
        confidence_map=None,
        if_update_confidence_map=True,
        if_screening=True,
        log_path='./train_log.txt',
        model_channel=[64],
        model_layers=20,
        optimizer_scheduler=True,
        dropout_alpha=0.9,
        data_augment=False
    ):
    B,C,H,W = Img_noisy.shape
    assert B == 1
    Img_noisy = Img_noisy.clamp(0,data_range)
    
    if confidence_map is None:
        m0 = torch.where((0.98*data_range<Img_noisy) | (0.02*data_range>Img_noisy), 1, 0)
        s = m0.float().sum(dim=1, keepdim=True)
        s = s / s.max()
        m1 = (s > 0.8*B)
        m0 = 1 - m0
        
        y0, y1 = image_filters(Img_noisy)
        m2 = (y0 - Img_noisy) < 0.2
        m3 = (y1 - Img_noisy) < 0.2
        
        confidence_map = (m3 & m2 & (m0 | m1)).float()
        
    screening_map = torch.ones_like(Img_noisy)
    pixel_numer = np.prod(Img_noisy.shape)
    confidence_map_update = confidence_map.clone()
    
    model, optimizer = reset_model(C, model_channel, model_layers)
    if optimizer_scheduler:
        scheduler = MultiStepLR(optimizer, milestones=[int(0.8*num_iter)], gamma=0.1)
    
    losses = []
    psnrs = []
    
    # denoised = inference_bayesian(Img_noisy, model, confidence_map)
    # residual = (denoised - Img_noisy)
    # p0 = len(confidence_map.nonzero())/pixel_numer
    # m = torch.exp(-residual.pow(2) * 1000)
    # m = m / m.max()
    # m = m * p0 / (m * p0 + 2*(1-p0))
    # print(m.max(), m.min())
    # cv2.imwrite(f'./cache/confidence_map_0.png', 255*tensor2numpy(m).squeeze()[:,:,-3])
    # cv2.imwrite(f'./cache/noise_indicator_0.png', 255*tensor2numpy(confidence_map_update).squeeze()[:,:,-3])
    thre = 0.1
    p0 = 0.9
    save_temp_name = f'temp{time.time()}'
    
    cv2.imwrite(f'./cache/{save_temp_name}_confidence.png', 255*tensor2numpy(confidence_map).squeeze()[:,:,-3])
    
    if clean_image is not None:
        cv2.imwrite(f'./cache/{save_temp_name}_clean.png', 255*clean_image[:,:,-3])
    for i in tqdm(range(int(num_iter))):
        loss = 0
        # detect the confident map
        if (i > 400) and ((i + 1) % int(0.1*num_iter) == 0):
            denoised = inference_bayesian(Img_noisy, model, confidence_map)
            residual = (denoised - Img_noisy)
            if if_update_confidence_map: 
                confidence_map_update = torch.where(thre>residual.abs(), 1, 0)
                confidence_map_update = (confidence_map.byte() & confidence_map_update.byte()).byte()

            if if_screening:
                with torch.no_grad():
                    p0 = len(confidence_map_update.nonzero())/pixel_numer
                    m = torch.exp(-residual.pow(2) * 1000)
                    m = m / m.max()
                    m = m * p0 / (m * p0 + 2*(1-p0))
                    screening_map = m
                    
            cv2.imwrite(f'./cache/{save_temp_name}_confidence.png', 255*tensor2numpy(confidence_map_update).squeeze()[:,:,-3])
            
        model.train()
        for kk in range(len(model)):
            mask = confidence_map_update*(torch.rand_like(Img_noisy) < dropout_alpha).float()

            net_input, mask, conf, smap = Img_noisy, mask, confidence_map_update, screening_map
            
            noise = torch.randn_like(net_input) * 0.1 if data_augment else 0
            net_output = model[kk](net_input + noise, mask)
            
            residual = net_input - net_output - noise
            
            if (i > 400) and if_screening:
                residual = residual * smap
            
            loss = loss + ((conf-mask)*residual).pow(2).sum().sqrt()
            cv2.imwrite(f'./cache/{save_temp_name}.png', 255*tensor2numpy(net_output).squeeze()[:,:,-3])
            
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if optimizer_scheduler:
            scheduler.step()
        
        if ((i+1) % print_epoch == 0):
            denoised = inference_bayesian(Img_noisy, model, confidence_map_update)
            if clean_image is not None:
                psnr_i = psnr(tensor2numpy(denoised)[0], clean_image, data_range=data_range)
                text_log = '( %d ) Loss, %.2f, min: %.2f, max: %.2f, PSNR: %.2f \n'%(i, loss.item(), denoised.min(), denoised.max(), psnr_i)
                print(text_log)
                psnrs.append(psnr_i)
            else:
                text_log = '( %d ) Loss: %.5f, min: %.2f, max: %.2f \n'%(i, round(loss.item(), 5), denoised.min(), denoised.max())
                print(text_log)
            with open(log_path, 'a') as f:
                f.writelines(text_log)
            
    denoised = inference_bayesian(Img_noisy, model, confidence_map_update)
    return denoised, confidence_map_update, screening_map, model, losses, psnrs