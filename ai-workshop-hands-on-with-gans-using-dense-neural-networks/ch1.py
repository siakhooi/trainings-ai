import os
import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm

#   ____ _                 _              _
#  / ___| |__   __ _ _ __ | |_ ___ _ __  / |
# | |   | '_ \ / _` | '_ \| __/ _ \ '__| | |
# | |___| | | | (_| | |_) | ||  __/ |    | |
#  \____|_| |_|\__,_| .__/ \__\___|_|    |_|
#                   |_|

if not os.path.exists('data'):
    os.makedirs('data')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean =  [0.5], std = [0.5]
    )
])

transform


fmnist_data = torchvision.datasets.FashionMNIST(
    root = 'data/', train = True, transform = transform, download = True
)



batch_size = 16

fmnist_dl = torch.utils.data.DataLoader(
    dataset = fmnist_data, batch_size = batch_size,
    shuffle = True, drop_last = True
)

fmnist_dl


print(len(fmnist_dl))



dataiter = iter(fmnist_dl)

images, labels = next(dataiter)



print(images.shape, labels.shape)



classes = [
    't_shirt/top',
    'trouser',
    'pullover',
    'dress',
    'coat',
    'sandal',
    'shirt',
    'sneaker',
    'bag',
    'ankle_boots'
]


import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

index=1
def display(img):
    global index

    # unnormalize
    img = img / 2 + 0.5

    img = img.permute(1, 2, 0)

    npimg = img.numpy()

    plt.imshow(npimg)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig(f'ch1-fmnist-{index}.png')
    index += 1
    plt.close()


display(make_grid(images, 4))

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))
