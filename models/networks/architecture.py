"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import functools
import numpy as np


class ASAPNetsResnetBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(True), kernel_size=3):
        super().__init__()
        self.conv_block = nn.Sequential(
            (nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1)),
            activation
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class ASAPNetsBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(True), kernel_size=3, reflection_pad=False, replicate_pad=False):
        super().__init__()
        padw = 1
        if reflection_pad:
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(padw),
                                            (nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )

        elif replicate_pad:
            self.conv_block = nn.Sequential(nn.ReplicationPad2d(padw),
                                            (nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )

        else:
            self.conv_block = nn.Sequential((nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padw)),
                                            activation
                                            )
    def forward(self, x):
        out = self.conv_block(x)
        return out


class ASAPNetsGradBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.ReLU(True), kernel_size=3, reflection_pad=False):
        super().__init__()
        padw = 1
        if reflection_pad:
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(padw),
                                            (nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0)),
                                            activation
                                            )
        else:
            self.conv_block = nn.Sequential((nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padw)),
                                            activation
                                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class MySeparableBilinearDownsample(torch.nn.Module):
    def __init__(self, stride, channels, use_gpu):
        super().__init__()
        self.stride = stride
        self.channels = channels

        # create tent kernel
        kernel = np.arange(1,2*stride+1,2) # ramp up
        kernel = np.concatenate((kernel,kernel[::-1])) # reflect it and concatenate
        if use_gpu:
            kernel = torch.Tensor(kernel/np.sum(kernel)).to(device='cuda') # normalize
        else:
            kernel = torch.Tensor(kernel / np.sum(kernel))
        self.register_buffer('kernel_horz', kernel[None,None,None,:].repeat((self.channels,1,1,1)))
        self.register_buffer('kernel_vert', kernel[None,None,:,None].repeat((self.channels,1,1,1)))

        self.refl = nn.ReflectionPad2d(int(stride/2))#nn.ReflectionPad2d(int(stride/2))

    def forward(self, input):
        return F.conv2d(F.conv2d(self.refl(input), self.kernel_horz, stride=(1,self.stride), groups=self.channels),
                    self.kernel_vert, stride=(self.stride,1), groups=self.channels)
