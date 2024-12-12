import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import matplotlib.pyplot as plt


# from google.colab import drive

# drive.mount("/content/drive")


#!unzip '/content/drive/MyDrive/ai_workshop_dcgans/anime_classification/anime_images.zip' -d'/content/images'
# !rm -rf '/content/images/__MACOSX'



device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")



batch_size = 64
image_size = 64

# Create the dataset
anime_faces_dataset = dset.ImageFolder(root = 'anime_classification',
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

# Create the dataloader
train_dl = torch.utils.data.DataLoader(
    anime_faces_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2,
    drop_last = True
)

len(train_dl)


real_batch = next(iter(train_dl))

plt.figure(figsize = (8 , 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(make_grid(real_batch[0].to(device), padding = 2, normalize = True).cpu(), (1, 2, 0)))
plt.savefig('ch3-3_training_images.png')
