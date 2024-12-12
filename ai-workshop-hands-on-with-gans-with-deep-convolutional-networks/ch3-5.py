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
plt.savefig('ch3-5-training_images.jpg')

#   ____     _____       _  _
#  / ___|   |___ /      | || |
# | |   _____ |_ \ _____| || |_
# | |__|_____|__) |_____|__   _|
#  \____|   |____/         |_|
#


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Number of GPUs
ngpu = 1



device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

device


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels = nz,
                out_channels = ngf * 8,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(
                in_channels = ngf * 8,
                out_channels = ngf * 4,
                kernel_size = 4,
                stride = 2, padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(
                in_channels = ngf * 4,
                out_channels = ngf * 2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(
                in_channels = ngf * 2,
                out_channels = ngf,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(
                in_channels = ngf,
                out_channels = nc,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.Tanh()

            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, x):
        return self.net(x)



netG = Generator().to(device)

netG.apply(weights_init)

print(netG)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(
                in_channels = nc,
                out_channels = ndf,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.LeakyReLU(0.2, inplace = True),

            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(
                in_channels = ndf,
                out_channels = ndf * 2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),

            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(
                in_channels = ndf * 2,
                out_channels = ndf * 4,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),

            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(
                in_channels = ndf * 4,
                out_channels = ndf * 8,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),

            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(
                in_channels = ndf * 8,
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



netD = Discriminator().to(device)

netD.apply(weights_init)

print(netD)


#   ____     _____      ____
#  / ___|   |___ /     | ___|
# | |   _____ |_ \ ____|___ \
# | |__|_____|__) |_____|__) |
#  \____|   |____/     |____/
#


z = torch.randn(64, 100, 1, 1, device = device)

# Generator network output
sample_gen_output = netG(z)

sample_gen_output

print(sample_gen_output.shape)


plt.imshow(np.transpose(make_grid(sample_gen_output[0].to(device), padding = 2, normalize = True).cpu(),(1, 2, 0)))
plt.savefig('ch3-5-1-sample_gen_output.jpg')

plt.imshow(np.transpose(make_grid(sample_gen_output.to(device), padding = 2, normalize = True).cpu(),(1, 2, 0)))
plt.savefig('ch3-5-2-sample_gen_output.jpg')


netD.eval()

with torch.no_grad():
    prediction = netD(sample_gen_output)

print(prediction[:10])

