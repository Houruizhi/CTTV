import math
import numpy as np

import torch
import torch.nn.functional as F

def nabla_x(x):
    '''
    x: (h,w,c)
    '''
    y = torch.zeros_like(x)
    y[:,1:,...] = x[:,1:,...] - x[:,:-1,...]
    y[:,0,...] = x[:,0,...] - x[:,-1,...]
    return y
    
def nabla_y(x):
    '''
    x: (h,w,c)
    '''
    y = torch.zeros_like(x)
    y[1:,:,...] = x[1:,:,...] - x[:-1,:,...]
    y[0,:,...] = x[0,:,...] - x[-1,:,...]
    return y

def nablat_x(x):
    '''
    x: (h,w,c)
    '''
    y = torch.zeros_like(x)
    y[:,:-1,...] = x[:,:-1,...] - x[:,1:,...]
    y[:,-1,...] = x[:,-1,...] - x[:,0,...] 
    return y

def nablat_y(x):
    '''
    x: (h,w,c)
    '''
    y = torch.zeros_like(x)
    y[:-1,:,...] = x[:-1,:,...] - x[1:,:,...]
    y[-1,:,...] = x[-1,:,...] - x[0,:,...] 
    return y

def shrink2_hard(x,y,lam):
    
    s = (x.pow(2) + y.pow(2)).sqrt().clamp(1e-4)
    sel = (s>lam)
    xs = x*sel/s
    ys = y*sel/s
    return xs, ys

def shrink2_MCP(x,y,lam,gamma):

    mu = gamma*lam
    s = (x.pow(2) + y.pow(2)).sqrt().clamp(1e-4)

    if lam < mu:
        # x == 0 when s<=lam
        M2 = ((s>=lam) & (s<mu))
        M3 = (s>=mu)
        xs = mu/(mu-lam)*(s-lam)*M2*x/s+x*M3
        ys = mu/(mu-lam)*(s-lam)*M2*y/s+y*M3
    else:
        xs, ys = shrink2_hard(x,y,math.sqrt(lam*mu))
    return xs, ys

def shrink21(x,y,lam):
    s = (x.pow(2) + y.pow(2)).sqrt().clamp(1e-4)
    s_thre = (s-lam).clamp(0)
    xs = x*s_thre/s
    ys = y*s_thre/s
    return xs, ys

def shrink1(x, lam):
    x_sign = x.sign()
    xs = torch.clamp(x.abs()- lam, 0)
    return xs*x_sign

def shrink1_huber(x, lam, mu=0.1):
    x_sign = x.sign()
    x_abs = x.abs()
    lam_plus_mu = mu+lam
    M = (x_abs > lam_plus_mu).byte()
    xs = M*x_abs + (1 - M)*(lam_plus_mu)
    return x + (lam/lam_plus_mu)*((1-lam/xs)*x-x)

def shrink1_truncted(x, lam, beta=5):
    x_sign = x.sign()
    xs = x.abs()
    M_truncted = (xs < beta*lam).byte()
    xs = M_truncted*torch.clamp(xs - lam, 0) + (1 - M_truncted)*xs
    return xs*x_sign

def shrink2_1_truncted(x,y,lam,gamma):

    mu = gamma*lam
    s = (x.pow(2) + y.pow(2)).sqrt().clamp(1e-4)

    if lam < mu:
        # x == 0 when s<=lam
        M2 = ((s>=lam) & (s<mu))
        M3 = (s>=mu)
        xs = (s-lam)*M2*x/s+x*M3
        ys = (s-lam)*M2*y/s+y*M3
    else:
        xs, ys = shrink2_hard(x,y,math.sqrt(lam*mu))
    return xs, ys

def shrink1_MCP(x,lam,gamma):
    mu = gamma*lam
    xs = x.abs()
    s = x.sign()
    if lam < mu:
        # x == 0 when xs<=lam
        M2 = ((xs>=lam) & (xs<mu))
        M3 = (xs>=mu)
        xs = mu/(mu-lam)*(xs-lam)*M2+xs*M3
    else:
        xs = xs.clamp(math.sqrt(lam*mu))
    return xs*s

def shrink1_MCP_veclam(x,lam,gamma):
    mu = gamma*lam
    M_ = (lam > mu).byte()
    xs = x.abs()
    s = x.sign()
    # x == 0 when xs<=lam
    M2 = ((xs>=lam) & (xs<mu))
    M3 = (xs>=mu)
    xs1 = mu/(1e-5+mu-lam)*(xs-lam)*M2+xs*M3
    return (1-M_)*xs1*s + M_*xs*(xs>lam*mu).byte()

def shrink_log(x, gamma, w):
    abs_x = x.abs()
    sign_x = x.sign()
    p = 0.5 * sign_x * ( abs_x-w + ((abs_x+w)**2 - 4*gamma).sqrt() )
        
    p = p*(abs_x > 2*math.sqrt(gamma) - w).byte()
    return p