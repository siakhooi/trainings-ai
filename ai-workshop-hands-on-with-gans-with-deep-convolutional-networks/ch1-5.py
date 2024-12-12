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


### Defining filter
# * Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters.
# * Filters for sharpen, edge detection , blurr and emboss operations are below.


sharpen_kernel = [[[[0, -1, 0]],
                   [[-1, 5, -1]],
                   [[0, -1, 0]]]]


sharpen_filter = torch.Tensor(sharpen_kernel)

print(sharpen_filter.shape)


### Applying filter
# * torch.nn.functional.conv2d accepts custom filters as opposed to torch.nn.conv2d which uses the default kernel
# * F.conv2d requires a 4d tensor as input. Hence, the unsqueeze operation


conv_tensor = F.conv2d(img_tensor, sharpen_filter, padding=0)

print(conv_tensor.shape)

conv_img = conv_tensor[0, :, :, :]

print(conv_img.shape)

conv_img = conv_img.numpy().squeeze()

print(conv_img.shape)


plt.figure(figsize = (12, 6))

plt.imshow(conv_img)
plt.savefig('ch-1-5-1-car_sharpened.jpg')
plt.close()

pool = nn.MaxPool2d(2, 2)

print(pool)

pool_tensor = pool(conv_tensor)

print(pool_tensor.shape)

pool_img = pool_tensor[0, :, :, :]

print(pool_img.shape)

pool_img = pool_img.numpy().squeeze()

print(pool_img.shape)


plt.figure(figsize = (12, 6))

plt.imshow(pool_img)
plt.savefig('ch-1-5-2-car_sharpened_pooled.jpg')
plt.close()

