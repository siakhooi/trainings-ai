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

#  ____  _               _           _             _
# |  _ \(_)___  ___ _ __(_)_ __ ___ (_)_ __   __ _| |_ ___  _ __
# | | | | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | __/ _ \| '__|
# | |_| | \__ \ (__| |  | | | | | | | | | | | (_| | || (_) | |
# |____/|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|\__\___/|_|
#

# On training mode
netD.train()

# On eval mode
netG.eval()

num_epochs = 1

for epoch in tqdm(range(num_epochs)):
    for i, data in enumerate(fmnist_dl, 0):
        ## Train with all-real batch
        netD.zero_grad()

        # Format batch
        real_images = data[0].reshape(batch_size, 1, 784)

        real_labels = torch.full((batch_size,), real_label, dtype=torch.float)

        # Forward pass real batch through D
        output = netD(real_images).view(-1)

        # Discriminator to should classify real images as real
        # Maximize the probability of this
        errD_real = criterion(output, real_labels)

        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_x = output.mean().item()

        # Generate fake image batch with G
        noise = torch.randn(batch_size, 100)

        fake_images = netG(noise)
        fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float)

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

        # Update D
        d_optimizer.step()

        if i%100 == 0:
            print(f"""Epoch {epoch+0:01}: |
                      D_real Loss: {errD_real:.3f} |
                      D_fake Loss: {errD_fake:.3f} |
                      D_total Loss: {errD:.3f} |
                      Fake_score {D_G_z1:.3f}|
                      Real_score {D_x:.3f}""")

