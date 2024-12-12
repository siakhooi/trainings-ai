import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchvision import transforms
from torchvision.utils import  make_grid

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print(device)


batch_size = 16
image_size = 64

# Create the dataset
anime_faces_dataset = dset.ImageFolder(
    root = 'anime_classification/',
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))


len(anime_faces_dataset)


classes = anime_faces_dataset.classes

classes


train_set, test_set = torch.utils.data.random_split(anime_faces_dataset, [1640, 408])


trainloader = torch.utils.data.DataLoader(
    train_set, batch_size = batch_size,
    shuffle = True, num_workers = 2
)


testloader = torch.utils.data.DataLoader(
    test_set, batch_size = batch_size,
    shuffle = False, num_workers = 2
)


dataiter = iter(trainloader)

images, labels = next(dataiter)
images.shape



def display(img):
    img = img / 2 + 0.5
    img = img.permute(1, 2, 0)
    npimg = img.numpy()

    plt.imshow(npimg)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('ch-2-3-1.jpg')
    plt.close()

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

display(make_grid(images, 8))

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8, 16)))
