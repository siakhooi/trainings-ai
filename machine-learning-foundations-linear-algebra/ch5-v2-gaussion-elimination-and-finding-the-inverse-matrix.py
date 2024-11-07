import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)

Ainv = np.linalg.inv(A)
print(Ainv)

b = np.array([5, 11])
print(b)

x = np.dot(Ainv, b)
print(x)

y = np.dot(A, x)
print(y)
