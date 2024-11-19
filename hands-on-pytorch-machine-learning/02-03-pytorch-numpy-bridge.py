import torch


# simple operation rehash
u = torch.tensor([2, 4])
v = torch.tensor([1, 4])
w = u * v
print(w)
print(w.shape)
print(w.ndim)


# broadcasting rules
a = torch.ones(4, 3, 2)
b = a * torch.rand(3, 2)  # 3rd & 2nd dims identical to a, dim 1 absent
print(b)


c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print(c)


d = a * torch.rand(1, 2)  # 3rd dim identical to a, 2nd dim = 1
print(d)


# switch between ndarrays and PyTorch tensors
import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)


# reverse example of converting from PyTorch tensor into ndarray
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)


# using the same underlying memory
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
