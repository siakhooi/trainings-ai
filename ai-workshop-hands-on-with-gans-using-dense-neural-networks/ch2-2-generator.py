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

#   ____ _                 _              ____
#  / ___| |__   __ _ _ __ | |_ ___ _ __  |___ \
# | |   | '_ \ / _` | '_ \| __/ _ \ '__|   __) |
# | |___| | | | (_| | |_) | ||  __/ |     / __/
#  \____|_| |_|\__,_| .__/ \__\___|_|    |_____|
#                   |_|


# Defining Hyperparameters for Generator and Discriminator Network

latent_size = 100

# 28 x 28
image_size = 784


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

netG = Generator()

print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p = 0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p = 0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(p = 0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


netD = Discriminator()

print(netD)


z = torch.randn(64, latent_size)

sample_gen_output = netG(z)

sample_gen_output

sample_gen_output.shape

torch.min(sample_gen_output).item(), torch.max(sample_gen_output).item()


generated_images = sample_gen_output.reshape(64, 1, 28, 28)

plt.imshow(generated_images[0, 0, :, :].detach().numpy(), cmap = 'gray')
plt.savefig('ch2-gen-image.png')

generated_images.shape


display(make_grid(generated_images, nrow = 8 , pad_value = 1.0))


netD.eval()

with torch.no_grad():
    prediction = netD(generated_images.reshape(-1, 784))

print(prediction[:20])

netD.eval()

with torch.no_grad():
    prediction_real = netD(images.reshape(-1, 784))

print(prediction_real[:20])



criterion = nn.BCELoss()

criterion



import torch.optim as optim

lr = 0.0002

d_optimizer = optim.Adam(netD.parameters(), lr = lr)

g_optimizer = optim.Adam(netG.parameters(), lr = lr)


real_label = 1.

fake_label = 0.

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
    plt.savefig(f'ch2-fmnist-{index}.png')
    index += 1
    plt.close()

#   ____                           _
#  / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
# | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
# | |_| |  __/ | | |  __/ | | (_| | || (_) | |
#  \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|
#

netG = Generator()

nedD = Discriminator()

d_optimizer = optim.Adam(netD.parameters(), lr = lr)

g_optimizer = optim.Adam(netG.parameters(), lr = lr)

# On training mode
netG.train()

# On eval mode
netD.eval()

num_epochs = 5

for epoch in tqdm(range(num_epochs)):

    for i, data in enumerate(fmnist_dl, 0):

        # Generate fake image batch with G
        noise = torch.randn(batch_size, 100)

        fake_images = netG(noise)
        real_labels = torch.full((batch_size,), real_label, dtype=torch.float)

        netG.zero_grad()

        output = netD(fake_images).view(-1)

        # For the generator, the discriminator to should classify fake images as real
        # Maximize the probability of this
        errG = criterion(output, real_labels)

        # Calculate gradients for G
        errG.backward()

        D_G_z2 = output.mean().item()

        # Update G
        g_optimizer.step()

        if i%100 == 0:
            print(f"""Epoch {epoch+0:01}: |
                      G_Loss: {errG:.3f} |
                      Fake_score_after_D_update: {D_G_z2:.3f}""")

            display_image_grid(
                epoch,i, fake_images.reshape(batch_size, 1, 28, 28).detach().cpu(), nrow = 8
            )

