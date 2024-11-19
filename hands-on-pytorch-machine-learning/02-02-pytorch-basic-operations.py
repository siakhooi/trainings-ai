import torch


# create 2 tensors
u = torch.tensor([1.0, 2.0])
v = torch.tensor([3.0, 4.0])

print(u)
print(v)


# perform some addition and substraction
print(u + v)
print(u - v)


# try some common tensor functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))


# reduction function using mean
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(x)
print(x.mean())

# more reduction function using min and max
print(x.max())
print(x.min())


# trigonometric functions and their inverses
import numpy as np

x = torch.tensor([0, np.pi / 2, np.pi])
print(x)
print(torch.sin(x))
print(torch.cos(x))


# get an evenly spaced list of numbers between a range
pi = torch.linspace(-np.pi/2, np.pi/2, steps=1000)
print(pi[:5])  # lower bound
print(pi[-5:]) # upper bound

sined = torch.sin(pi)
cosed = torch.cos(pi)
print(sined[0:5])
print(cosed[0:5])


# use matplotlib to visualize
import matplotlib.pyplot as plt

plt.plot(sined, label="sined")
plt.plot(cosed, label="cosed")
plt.legend()
plt.show()

plt.savefig('02-02-sined_cosed.png')
