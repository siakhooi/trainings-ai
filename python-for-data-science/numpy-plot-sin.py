import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
# %matplotlib inline
plt.plot(x, y)
plt.savefig("sin.png")
