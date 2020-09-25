import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from utils import Sigmoid
sigmoid = Sigmoid()

def Defense(x, in_channel, out_channel, kernel, a, b, clip_min_weight, 
                  clip_max_weight, clip_min_bias, clip_max_bias):
    weights = torch.randn(out_channel, in_channel, 
                      kernel, kernel).uniform_(clip_min_weight, clip_max_weight).to(device)
    bias = torch.randn(out_channel).uniform_(clip_min_bias , clip_max_bias).to(device)
    downsample = nn.Conv2d(in_channel, out_channel, kernel, stride=1, padding=1)   
    downsample.weight.data =  weights
    downsample.bias.data = bias    
    
    upsample = nn.ConvTranspose2d(out_channel, in_channel, kernel, stride=1, padding=1)       
    upsample.weight.data = weights
    upsample.bias.data = upsample.bias.data.uniform_(clip_min_bias, clip_max_bias).to(device)
    
    h = downsample(x)
    images = upsample(h) 
    images = sigmoid(images, a, b) 
    return images
