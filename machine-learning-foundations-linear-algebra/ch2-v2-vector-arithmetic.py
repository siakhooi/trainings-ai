import numpy as np

a = np.array([20, 40, 60])
b = np.array([10, 20, 30])
c = np.array([5, 10, 15, 20])

# Addition

plus = a + b
print(plus)
# a + c  # Error


# Subtraction

minus = a - b
print(minus)
# a - c  # Error

# Multiplication

multi = a * b
print(multi)

# Divide

div = a / b
print(div)

# Scalar

scalar = 5
list_a = [10, 11, 12, 13, 14, 15]
print(list_a)

list_as_array = np.array(list_a)
print(list_as_array)

l = scalar * list_a  # duplicated

print(l)

n = scalar * list_as_array  # element multiplied by scalar

print(n)
