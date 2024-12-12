import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print(torch.__version__)

img = Image.open('car.jpg')

print(img)


transforms = T.Compose([T.Resize(450),
                        T.ToTensor()])



img_tensor = transforms(img)

print(img_tensor)


print(img_tensor.shape)


img_tensor = img_tensor.unsqueeze(0)

print(img_tensor.shape)

