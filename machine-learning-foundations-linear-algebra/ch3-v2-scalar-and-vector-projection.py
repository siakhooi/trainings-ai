import numpy as np
from numpy import linalg as lng

a = np.array([10, 20, 30, 40, 50])
b = np.array([60, 70, 80, 90, 100])

len_a = lng.norm(a)

print(len_a)

vector_projection = (np.dot(a, b) / np.dot(b, b)) * b
print(vector_projection)
