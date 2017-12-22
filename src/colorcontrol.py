import os
import sys
import tqdm
import copy
import numpy as np
from PIL import Image
import logging

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from skimage import io
from skimage import transform
from skimage import color


import torchvision
from torchvision import transforms, datasets, models


from model import ImageTransformationNetwork, VGG16
from loss import FeatureReconstructionLoss, GramMatrix, StyleReconstructionLoss

image_dir = "../Images/"

source = io.imread(image_dir + 'test_image.jpg')
source = transform.resize(source,(256,256))	
source_yuv = color.rgb2yuv(source)
print(source_yuv.shape)

img = io.imread(image_dir + 'final_image.jpg')
img = transform.resize(img,(256,256))	
img_yuv = color.rgb2yuv(img)
print(img_yuv.shape)


img_yuv[:,:,0]= img_yuv[:,:,0] - img_yuv[:,:,0].mean()
img_yuv[:,:,0] = img_yuv[:,:,0] + source_yuv[:,:,0].mean()
source_yuv[:,:,0] = img_yuv[:,:,0]
new_img = color.yuv2rgb(source_yuv).clip(0,1)
print(type(new_img))
print(new_img.shape)
io.imsave(image_dir+'output.jpg',new_img)


