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

index=1
os.makedirs("output", exist_ok = True)

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
plt.savefig('ch3-6-training_images.jpg')

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



#   ____     _____        __
#  / ___|   |___ /       / /_
# | |   _____ |_ \ _____| '_ \
# | |__|_____|__) |_____| (_) |
#  \____|   |____/       \___/
#


# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device = device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0.5, 0.999))


def display_image_grid(epoch: int, step: int,  images: torch.Tensor, nrow: int):
    images = images / 2 + 0.5
    image_grid = make_grid(images, nrow)     # Images in a grid
    image_grid = image_grid.permute(1, 2, 0) # Move channel last
    image_grid = image_grid.cpu().numpy()    # To Numpy

    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    global index
    plt.savefig(f'output/ch3-6-{index}-epoch-{epoch}-step-{step}.jpg')
    index+=1
    plt.close()


G_losses = []
D_losses = []

real_score_list = []
fake_score_list = []

iters = 0
num_epochs = 50

fixed_noise = torch.randn(64, nz, 1, 1, device = device)

for epoch in tqdm(range(num_epochs)):
    # For each batch in the dataloader
    for i, data in enumerate(train_dl, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        real_images = data[0].to(device)
        real_labels = torch.full((batch_size,), real_label, dtype = torch.float, device = device)

        # Forward pass real batch through D
        output = netD(real_images).view(-1)

        # Discriminator to should classify real images as real
        # Maximize the probability of this
        errD_real = criterion(output, real_labels)

        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_x = output.mean().item()

        # Generate fake image batch with G
        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        fake_images = netG(noise)
        fake_labels = torch.full((batch_size,), fake_label, dtype = torch.float, device = device)

        # Classify all fake batch with D
        output = netD(fake_images.detach()).view(-1)

        # Discriminator to should classify fake images as fake
        # Maximize the probability of this
        errD_fake = criterion(output, fake_labels)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()

        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        optimizerD.step()

        ###########################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        netG.zero_grad()

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake_images).view(-1)

        # Maximize the probability that the discriminator categorizes fake
        # images as real
        errG = criterion(output, real_labels)

        # Calculate gradients for G
        errG.backward()

        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        if (i % 200 == 0) or ((epoch == num_epochs-1) and (i == len(train_dl) - 1)):
            print(f"""Epoch {epoch+0:01}: |
                      Step: {i} |
                      D_real Loss: {errD_real:.3f} |
                      D_fake Loss: {errD_fake:.3f} |
                      D_total Loss: {errD:.3f} |
                      G_Loss: {errG:.3f} |
                      Real_score {D_x:.3f} |
                      Fake_score {D_G_z1:.3f} |
                      Fake_score_after_D_update: {D_G_z2:.3f}
            """)

            # View the output of the GAN every so often
            with torch.no_grad():
                fake_images = netG(fixed_noise).reshape(batch_size, 3, 64, 64).detach().cpu()
                display_image_grid(epoch, i, fake_images, nrow = 8)

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        real_score_list.append(D_x)
        fake_score_list.append(D_G_z1)

