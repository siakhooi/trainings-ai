import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

index=1

img = Image.open('car.jpg')
transforms = T.Compose([T.Resize(450),
                        T.ToTensor()])
img_tensor = transforms(img)
img_tensor = img_tensor.unsqueeze(0)


def apply_kernel_and_show(img_tensor, kernel, title):
  print(title)

  filter = torch.Tensor(kernel)

  conv_tensor = F.conv2d(img_tensor, filter, padding=0)

  conv_img = conv_tensor[0, :, :, :]
  conv_img = conv_img.numpy().squeeze()

  pool = nn.MaxPool2d(2, 2)

  pool_tensor = pool(conv_tensor)

  pool_img = pool_tensor[0, :, :, :]
  pool_img = pool_img.numpy().squeeze()

  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.set_size_inches(16, 6)

  fig.suptitle('Convolutional output and Pooling output')

  ax1.imshow(conv_img)
  ax2.imshow(pool_img)
  global index
  plt.savefig(f'ch-1-6-{index}-{title}.jpg')
  index+=1


vertical_edge_kernel = [[[[-1, 0, 1]],
                       [[-1, 0, 1]],
                       [[-1, 0, 1]]]]

apply_kernel_and_show(img_tensor, vertical_edge_kernel, "vertical edge")

horizontal_edge_kernel = [[[[-1, -1, -1]],
                           [[0, 0, 0]],
                           [[1, 1, 1]]]]

apply_kernel_and_show(img_tensor, horizontal_edge_kernel, "horizontal edge")

gaussian_blur_kernel = [[[[1/16, 1/8, 1/16]],
                           [[1/8, 1/4, 1/8]],
                           [[1/16, 1/8, 1/16]]]]

apply_kernel_and_show(img_tensor, gaussian_blur_kernel, "gaussian blur")

emboss_kernel = [[[[-2, -1, 0]],
                  [[-1, 1, 1]],
                  [[0, 1, 2]]]]

apply_kernel_and_show(img_tensor, emboss_kernel, "emboss")

