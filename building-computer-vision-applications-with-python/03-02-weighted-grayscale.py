index = 0

def savepic(label):
    global index
    plt.savefig(f"03-02-{index}-{label}.png")
    index = index + 1
    plt.close()


# Weighted Grayscale
# Average grayscale doesn't usually look very natural.
# A more natural alternative consists in assigning different weights to Red, Green, and Blue, based on luminance or human perception.

# A popular distribution is:
# * Red: 0.299
# * Green: 0.587
# * Blue: 0.114


# Let's open the Playspace image again

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
# %matplotlib inline
from EC_CV import *
from matplotlib import rcParams

plt.rcParams['figure.figsize'] = (8, 8)

toys = adapt_PNG(plt.imread('playspace.png'))
plt.axis("off")
plt.imshow(toys)

savepic('playspace-original')


rcParams['figure.figsize'] = (20, 8)

# Calculate regular average and weighted average
toys_avg = np.dot(toys[...,:3], [1/3,1/3,1/3])
toys_wgt = np.dot(toys[...,:3], [0.299,0.587,0.114])

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(toys_avg, cmap='gray')
ax[1].imshow(toys_wgt, cmap='gray')

savepic('playspace-average-vs-weighted')


# Now let's try it on another picture

fruit = plt.imread('fruit.jpg')
plt.imshow(fruit)
savepic('fruit-original')


fruit_avg = np.dot(fruit[...,:3], [1/3,1/3,1/3])
fruit_wgt = np.dot(fruit[...,:3], [0.299,0.587,0.114])

# figure size in inches
rcParams['figure.figsize'] = 20, 8

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(fruit_avg, cmap='gray')
ax[1].imshow(fruit_wgt, cmap='gray')


savepic('fruit-average-vs-weighted')
