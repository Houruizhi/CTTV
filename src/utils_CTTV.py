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
from scipy.io import savemat
from .tv_op import nabla_x, nabla_y

from skimage.metrics import peak_signal_noise_ratio as psnr

def inference_bayesian(Y, V, model, alpha=0.9, N=30):
    model.eval()
    U = 0
    for _ in range(N):
        with torch.no_grad():
            mask1 = (torch.rand_like(Y) < alpha).float()
            U += model[0](Y, mask1)
    U = U / N      
    return U.permute(0,2,3,1).squeeze(0).cpu()

def reset_model(C,C_out,model_channel=64,model_layers=20):
    from src.models.dncnn_dropout import DnCNN as DnCNN_dropout
    model = nn.ModuleList()
    model.append(DnCNN_dropout(C,C_out,model_layers,model_channel,kernel_size=3,norm_type='bn').cuda())
    model.train()
    
    optimizer = Adam([{'params': model.parameters()}], lr=1e-3)
    return model, optimizer

def forward_once(model, optimizer, Y, U, V, mu=1, Z=1, Z2=1):
    model.train()
    V = V.cuda()
    
    U_ = U.permute(0,2,3,1)@V.transpose(1,0)
    U_ = U_.permute(0,3,1,2)
    
    n,c,h,w = U.shape
    # std_ = U.view(n,c,-1).std(-1).view(n,c,1,1)
    noise = torch.randn_like(Y) * 0.1
    inputs = Y

    noise_U = torch.randn_like(U) * 0.1
    
    mask = (torch.rand_like(Y) < 0.9).float()
    net_outputU = model[0](inputs+noise,mask)+noise_U
    net_outputX = (net_outputU.permute(0,2,3,1))@V.transpose(1,0)
    net_outputX = net_outputX.permute(0,3,1,2)
    
    residual = Z*(Y - net_outputX)
    
    loss = 0
    loss = loss + ((1-mask)*residual).pow(2).sum().sqrt()
    loss = loss + mu*(Z2*(net_outputU-U)).pow(2).sum().sqrt()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model, optimizer

from .tv_op import *
def CTTV(Y, r=13, tau1=0.0012, tau2=0.0012, mu=1000, eta=1.1, X_ori=None, nonconvex=True, nonlinear=True, MAX_ITER=80):
    '''
    Y: (h,w,c)
    '''
    h,w,p = Y.shape

    sizeU = [h,w,r]
    sizeX = [h,w,p]
    sizeVec = [h*w,p]
    sizeVecU = [h*w,r]

    D = Y.reshape(sizeVec)
    u,s,v = torch.svd(D)
    s = torch.diag(s)
    U = u[:,:r]@s[:r,:r]
    V = v[:,:r]

    X = (U@V.transpose(1,0)).reshape(*sizeX)
    U = U.reshape(*sizeU)
    S = torch.zeros_like(X)
    E = torch.zeros_like(X)

    uker1 = torch.zeros_like(U)
    uker1[...,0,0,0] = -1
    uker1[...,-1,0,0] = 1
    uker2 = torch.zeros_like(U)
    uker2[...,0,0,0] = -1
    uker2[...,0,-1,0] = 1

    uker1 = torch.fft.fftn(torch.complex(uker1, torch.zeros_like(uker1)))
    uker2 = torch.fft.fftn(torch.complex(uker2, torch.zeros_like(uker2)))
    uker = uker1.abs().pow(2) + uker2.abs().pow(2)

    M1 = torch.zeros_like(U)  # multiplier for Dx_U-G1
    M2 = torch.zeros_like(U)  # multiplier for Dy_U-G2
    M3 = torch.zeros_like(Y)  # multiplier for D-UV^T-E
    
    normD = Y.pow(2).sum().sqrt()
    mu = mu/normD
    
    if nonlinear:
        model, optimizer = reset_model(p,r,2*r,17)
        
    tau = np.array([tau1, tau2])  * math.sqrt(Y.shape[0]*Y.shape[1])
    eta = 1.1
    Y_train_CNN = (Y).reshape(*sizeX).unsqueeze(0).permute(0,3,1,2).cuda()
    
    PSNRs = []
    U0 = U
    sigmaU = 0.2
    for idx in range(MAX_ITER):
        if nonlinear:
            U_train = (U).unsqueeze(0).permute(0,3,1,2).cuda()
            # num_iter = 50 #if mu < 5 else 20
            num_iter = 20
            for _ in tqdm(range(num_iter)):
                # model, optimizer = forward_once(model, optimizer, Y_train_CNN, U_train, V, mu, screening_map, screening_map2)
                model, optimizer = forward_once(model, optimizer, Y_train_CNN, U_train, V, mu)
            U_ = inference_bayesian(Y_train_CNN, V, model, N=30)
            sigmaU *= 0.8
        else:
            U_ = U
        
        R = Y-E-S+M3/mu
        
        res = (R + 1e-2*X).reshape(*sizeVec)
        temp = (res@V).reshape(*sizeU) + sigmaU*(U_)
        
        d1 = nabla_x(U)+M1/mu
        d2 = nabla_y(U)+M2/mu
        if nonconvex:
            G1 = shrink1_MCP(d1, tau[0]/mu, 5)
            G2 = shrink1_MCP(d2, tau[0]/mu, 5)
        else:
            G1 = shrink1(d1, tau[0]/mu)
            G2 = shrink1(d2, tau[0]/mu)
            
        diffT_p = nablat_x(G1-M1/mu) + nablat_y(G2-M2/mu)
        numer1 = diffT_p + temp

        numer1 = torch.fft.fftn(torch.complex(numer1, torch.zeros_like(numer1)))
        U = torch.fft.ifftn(numer1/ (uker + 1 + 1e-2 + sigmaU)).real

        U_vec = U.reshape(*sizeVecU)
        u,_,vh = torch.linalg.svd(res.transpose(1,0)@U_vec,full_matrices=False)
        V = u@vh

        X = (U_vec@V.transpose(1,0)).reshape(*sizeX)
        
        residual_Y_X = Y-X+M3/mu

        E = ((residual_Y_X-S))/(1+1e3/mu)
        
        S = shrink1(residual_Y_X-E, tau[1]/(mu))
        leq1 = nabla_x(U)-G1
        leq2 = nabla_y(U)-G2
        leq3 = Y-X-E-S

        M1 = M1 + mu*leq1
        M2 = M2 + mu*leq2
        M3 = M3 + mu*leq3
        mu = min(1e6,mu*eta)
  
        if X_ori is not None:
            PSNR0 = 0
            for c in range(X.shape[-1]):
                PSNR0 += psnr(X[:,:,c].clip(0,1).numpy(), X_ori[:,:,c].clip(0,1), data_range=1)
            
            print(idx, mu, PSNR0 / X.shape[-1])
            PSNRs.append(PSNR0 / X.shape[-1])

    return X,U,V,PSNRs