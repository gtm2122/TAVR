import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models import *
from data_utils import cache_loader
import torch.optim as optim
from losses import DICELossMultiClass 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
from train_main import *



image = torch.zeros((1,1,256,256,256))
mask = torch.zeros((1,1,256,256,256))


mask = torch.cat((mask,1-mask),dim=1)
image, mask = image.cuda(), mask.cuda()

image, mask = Variable(image), Variable(mask)
#print(image.size())
print(image.size())
print(mask.size())
#model  = UnetGenerator_3d(in_dim=1,out_dim=2,num_filter=1,crop=False,smax=False).cuda()

in_dim = 1
out_dim = 2
act_fn = nn.ReLU(inplace=True)
model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    ).cuda()
optimizer = optim.SGD(model.parameters(),lr=0.1)
optimizer.zero_grad()

output = model(image.contiguous())
#print(image.size())
#         print(output.size())
#         print(mask.size())



criterion = torch.nn.L1Loss()#DICELossMultiClass()

optimizer.zero_grad()

output = model(image.contiguous())
print(output.size())
#print(image.size())
#         print(output.size())
#         print(mask.size())
#         print(output.size())
#         print(mask.size())

loss = criterion(output.contiguous(), mask.contiguous())
#         print(loss.size())
#         print(loss)
loss = loss.contiguous()
# e_loss+=loss.item()
#print(len(train_loader))
#         print(c)
# if c in l:
#     print(100*c/len(train_loader))

loss.backward()
optimizer.step()
