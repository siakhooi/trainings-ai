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
    plt.savefig(f'output/ch3-2-fmnist-{index}.png')
    index += 1
    plt.close()


#   ____ _                 _              _____
#  / ___| |__   __ _ _ __ | |_ ___ _ __  |___ /
# | |   | '_ \ / _` | '_ \| __/ _ \ '__|   |_ \
# | |___| | | | (_| | |_) | ||  __/ |     ___) |
#  \____|_| |_|\__,_| .__/ \__\___|_|    |____/
#                   |_|

netG = Generator()

nedD = Discriminator()

d_optimizer = optim.Adam(netD.parameters(), lr = lr)

g_optimizer = optim.Adam(netG.parameters(), lr = lr)


avg_G_losses = []
avg_D_losses = []
avg_real_score_list = []
avg_fake_score_list = []

num_epochs = 40

fixed_noise = torch.randn(batch_size, 100)

netD.train()
netG.train()


for epoch in tqdm(range(num_epochs)):

    G_losses = []
    D_losses = []

    real_score_list=[]
    fake_score_list=[]

    for i, data in enumerate(fmnist_dl, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        netD.zero_grad()

        real_images = data[0].reshape(batch_size, 1, 784)
        real_labels = torch.full((batch_size,), real_label, dtype = torch.float)

        output = netD(real_images).view(-1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()

        D_x = output.mean().item()

        # Generate fake image batch with G
        noise = torch.randn(batch_size, 100)
        fake_images = netG(noise)
        fake_labels = torch.full((batch_size,), fake_label, dtype = torch.float)

        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()

        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        d_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        output = netD(fake_images).view(-1)
        errG = criterion(output, real_labels)

        errG.backward()

        D_G_z2 = output.mean().item()
        g_optimizer.step()

        if (i % 200 == 0) or ((epoch == num_epochs-1) and (i == len(fmnist_dl) - 1)):
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

            with torch.no_grad():
                fake_images = netG(fixed_noise).reshape(batch_size, 1, 28, 28).detach().cpu()
                display_image_grid(epoch, i, fake_images, nrow = 8)

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        real_score_list.append(D_x)
        fake_score_list.append(D_G_z1)

    avg_G_losses.append(np.mean(G_losses))
    avg_D_losses.append(np.mean(D_losses))
    avg_real_score_list.append(np.mean(real_score_list))
    avg_fake_score_list.append(np.mean(fake_score_list))




fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8))

ax1.plot(avg_G_losses, label = 'Generator loss', alpha = 0.5)
ax1.plot(avg_D_losses, label = 'Discriminator loss', alpha = 0.5)
ax1.legend()

ax1.set_title('Training Losses')

ax2.plot(avg_real_score_list, label = 'Real score', alpha = 0.5)
ax2.plot(avg_fake_score_list, label = 'Fake score', alpha = 0.5)
ax2.set_title('Accuracy Scores')

ax2.legend()

plt.savefig('ch3-2-avg-loss-and-accuracy.png')
plt.close()
