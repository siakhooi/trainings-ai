import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# dot

ab = np.dot(a, b)

print(ab)

c = np.array([11, 12, 13, 14, 15])

ba = np.dot(b, a)

print(ba)

first_result = np.dot(a, b + c)
print(first_result)

second_result = np.dot(a, b) + np.dot(a, c)

print(second_result)  # same with first_result
