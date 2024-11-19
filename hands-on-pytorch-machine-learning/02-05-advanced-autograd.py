# require gradient on a variable

import torch
from torch.autograd import Variable


x = torch.randn(3)
x = Variable(x, requires_grad=True)
print(x)


y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)


gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)


print(x.grad)


# Fronzen parameters


from torch import nn, optim
import torchvision


import ssl

ssl._create_default_https_context = ssl._create_unverified_context


model = torchvision.models.resnet18(pretrained=True)


# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


print(optimizer)
