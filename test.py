import torch.nn as nn
import torch
import torchvision.utils as vutils
import numpy as np

a = np.random.rand(30,3,64,64)
print(a.shape)
b = a.transpose([0,2,3,1])
print(b.shape)

def fuc():
    return 1,2

c = fuc()

print(c)

vutils.save_image()