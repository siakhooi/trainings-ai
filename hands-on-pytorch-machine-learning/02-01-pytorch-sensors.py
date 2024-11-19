import torch


# create some tensors
t0 = torch.tensor(1000)  # 0-D tensor
t1 = torch.tensor([9, 8, 7, 6])  # 1-D tensor
t2 = torch.tensor([[1, 2, 3], [7, 5, 3]])  # 2-D tensor

print(t0)
print(t1)
print(t2)


# set datatype for the tensor
a = torch.ones((2, 3), dtype=torch.int16)
print(a)


""" Tensor data types
torch.bool
torch.int8
torch.uint8
torch.int16
torch.int32
torch.int64
torch.half
torch.float
torch.double
torch.bfloat
"""


# get the tensor size
print(a.size())
print(a.shape)


# get the tensor dimension
print(a.ndimension())
print(a.ndim)
