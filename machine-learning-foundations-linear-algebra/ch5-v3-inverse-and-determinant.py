import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)

det = np.linalg.det(A)
print(det)

i = np.linalg.inv(A)
print(i)

B = np.array([[3, 1], [6, 2]])
print(B)

d = np.linalg.det(B)
print(d)

i = np.linalg.inv(B) # determinant=0, singular matrix, no inverse
print(i)
