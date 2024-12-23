import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.register_buffer('weight_maskUpdater', torch.ones(self.out_channels, self.in_channels, 3, 3))
        self.slide_winsize = np.prod(self.weight_maskUpdater.shape[1:])

        self.update_mask = None
        self.mask_ratio = None

        std_ = 1/math.sqrt(self.in_channels*self.kernel_size[0]*self.kernel_size[0])
        self.weight.data = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size[0],self.kernel_size[1]).normal_(mean=0, std=std_))
        
    def forward(self, input, mask_in=None):
        if mask_in is not None:
            with torch.no_grad():
                self.update_mask = F.conv2d(mask_in, self.weight_maskUpdater, padding=1, groups=1)
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
            
        output = super(PartialConv2d, self).forward(torch.mul(input, mask_in) if mask_in is not None else input)
        
        if mask_in is None:
            self.update_mask = torch.ones_like(output)
            self.mask_ratio = self.slide_winsize
        else:  
            output = torch.mul(output, self.mask_ratio)

        return output, self.update_mask

def get_norm_layer(norm_type):
    if norm_type == 'bn':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == None:
        norm_layer = nn.Identity
    else:
        raise NotImplementedError(f'{norm_type} is not implemented')
    return norm_layer

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers, num_features, kernel_size=3, norm_type='bn'):
        super(DnCNN, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        kernel_size = kernel_size
        padding = kernel_size // 2
        features = num_features
        in_channels = in_channels
        layers = []
        layers.append(PartialConv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_of_layers-2):
            layers.append(PartialConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(norm_layer(features))
            layers.append(nn.ReLU(inplace=True))
        self.dncnn = nn.Sequential(*layers)
        
        self.tail = nn.Conv2d(features, out_channels, kernel_size=1)
        self.tail.weight.data = torch.ones(out_channels,features,1,1)
        self.tail.weight.data = self.tail.weight.data / self.tail.weight.data.sum()
        
    def forward(self, inputs, mask=None, return_features=False):
        x = inputs
        for m in self.dncnn:
            if isinstance(m, PartialConv2d):
                x, mask = m(x, mask)
            else:
                x = m(x)
                
        features = x
        x = self.tail(features)
        
        if return_features:
            return x, features
        else:
            return x