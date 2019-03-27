import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models import *

import torch.optim as optim

import torch.nn.functional as F
import matplotlib.pyplot as plt

import pickle
 

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

image = torch.zeros((1,1,256,256,256))
mask = torch.zeros((1,1,256,256,256))


mask = torch.cat((mask,1-mask),dim=1)
image, mask = image.cuda(), mask.cuda()

image, mask = Variable(image), Variable(mask)
#print(image.size())
print(image.size())
print(mask.size())

in_dim = 1
out_dim = 2
act_fn = nn.ReLU(inplace=True)

model = nn.Sequential(conv_block_3d(in_dim,out_dim,act_fn)).cuda()

optimizer = optim.SGD(model.parameters(),lr=0.1)
optimizer.zero_grad()

output = model(image.contiguous())


criterion = torch.nn.L1Loss()#DICELossMultiClass()

optimizer.zero_grad()

output = model(image.contiguous())
print(output.size())

loss = criterion(output.contiguous(), mask.contiguous())

loss = loss.contiguous()

loss.backward()
optimizer.step()
