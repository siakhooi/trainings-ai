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



batch_size = 16 #8
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

index=1


def display(img):
    img = img / 2 + 0.5
    img = img.permute(1, 2, 0)
    npimg = img.numpy()

    plt.imshow(npimg)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    global index
    plt.savefig(f'ch2-6-display-{index}.jpg')
    index+=1
    plt.close()

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

display(make_grid(images, 8))

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8, 16)))



# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002




def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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


criterion = nn.BCELoss()

optimizer = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))




for epoch in range(2):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = netD(inputs)

        loss = criterion(outputs.flatten(), labels.float())
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 20==0:    # print every 20 mini-batches
            print(f'[Epoch: {epoch + 1}, Step: {i + 1:5d}] loss: {running_loss/20:.3f}')
            running_loss = 0.0

print('Finished Training')



dataiter = iter(testloader)

images, labels = next(dataiter)

print('GroundTruth: ')

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
display(make_grid(images, 8))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8, 16)))



outputs = netD(images.to(device))



predicted = torch.round(outputs.flatten())

print('Predicted: ')

print(' '.join(f'{classes[predicted.long()[j]]:5s}' for j in range(8)))
display(make_grid(images, 8))
print(' '.join(f'{classes[predicted.long()[j]]:5s}' for j in range(8, 16)))



correct = 0
total = 0

with torch.no_grad():
    for data in testloader:

        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        outputs = netD(images)

        predicted = torch.round(outputs.flatten()).long()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')



# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

all_labels = []
all_predictions = []

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        all_labels.append(labels.cpu().numpy())

        outputs = netD(images)
        predictions = torch.round(outputs.flatten()).long()

        all_predictions.append(predictions.cpu().numpy())

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def show(image, title):

    plt.figure()

    plt.imshow(np.clip(image.permute(1, 2, 0).cpu().numpy(), 0 ,1))

    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict = {'size': 16})
    global index
    plt.savefig(f'ch2-6-show-{index}.jpg')
    index+=1
    plt.close()



for i, (images, labels) in enumerate(testloader):
    for j in range(len(all_predictions[i])):
      predicted_class = all_predictions[i][j]
      actual_class = all_labels[i][j]

      if predicted_class != actual_class:
        show(images[j], 'Model prediction {} (class {}), actual category {} (class {})'.format(
            classes[predicted_class], predicted_class,
            classes[actual_class], actual_class
        ))

