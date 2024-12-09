# Import the required libraries
# For this example we will use pytorch to manage the construction of the neural networks and the training
# torchvision is a module that is part of pytorch that supports vision datasets and it will be where we will source the mnist - handwritten digits - data

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils




# Setting a seed will determine which data elements are selected. To replicate results keep the same seed.
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



# This is a check if there is a gpu available for training. At the moment we are assuming that it is not available.
torch.cuda.is_available()



# Assuming the GPU is not available means we will set the device to cpu and set up some parameters
cudnn.benchmark = True
device = torch.device("cpu")
ngpu = 0
#This is the width of the latent space matrix
nz = 100
# This is the generator matrix shape
ngf = 64
# This is the descrimator matrix shape
ndf = 64
# This is the number of color channels - other datasets may have 3 if they are color
nc = 1
# The number of sample to process per pass
batch_size = 64
# the number of CPU workers to work on the dataset
workers = 4



dataset = dset.MNIST(root='data', download=True,
                      transform=transforms.Compose([
                          transforms.Resize(64),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),
                      ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))







# custom weights initialization called on netG and netD
# The weights will need to be initialised based on the layer type to some value before training. These could be imported from past training steps.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)





# This is the bulk of the neural network definition for the Generator.
# The init sets up the layers and connecting activation functions.
# The forward function processes the data through the layers
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)




# This is the bulk of the neural network definition for the Discrimator.
# The init sets up the layers and connecting activation functions.
# The forward function processes the data through the layers
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)





# Set the loss function from pytorches established modules
criterion = nn.BCELoss()

# Set up the initial noise of the latent space to sample from.
# Set the label of a real and fake sample to 0,1
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# Create the optimiser which will dynamically change the parameters of the learning function over time to imporve the training process
optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0005, betas=(0.5, 0.999))






# This is the engine of the code base - explicitly taking the objects created above
# (The generator, discrimator and the dataset) and connecting them together to learn.

for epoch in range(1):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        # Set the descrimator to forget any gradients.
        netD.zero_grad()
        # Get a sample of real handwritten digits and label them as 1 - all real
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
        # Pass the sample through the discrimator
        output = netD(real_cpu)
        # measure the error
        errD_real = criterion(output, label)
        # Calculate the gradients of each layer of the network
        errD_real.backward()
        # Get the average of the output across the batch
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        # pass the noise through the generator layers
        fake = netG(noise)
        # set the labels to all 0 - fake
        label.fill_(fake_label)
        # ask the discrimator to judge the fake images
        output = netD(fake.detach())
        # measure the error
        errD_fake = criterion(output, label)
        # Calculate the gradients
        errD_fake.backward()
        # Get the average output across the batch again
        D_G_z1 = output.mean().item()
        # Get the error
        errD = errD_real + errD_fake
        # Run the optimizer to update the weights
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # Set the gradients of the generator to zero
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # get the judgements from the discrimator of the generator output is fake
        output = netD(fake)
        # calculate the error
        errG = criterion(output, label)
        # update the gradients
        errG.backward()
        # Get the average of the output across the batch
        D_G_z2 = output.mean().item()
        # update the weights
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 1, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # every 100 steps save a real sample and a fake sample for comparison
        if i % 100 == 0:
            vutils.save_image(real_cpu,'real_samples.png',normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),'fake_samples_epoch_%03d.png' % epoch, normalize=True)



