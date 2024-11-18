index = 0

def savepic(label):
    global index
    plt.savefig(f"04-07-{index}-{label}.png")
    index = index + 1
    plt.close()

# Don't modify this cell, just run it.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams['figure.figsize'] = (10, 8)



# Task #1: Print a color image
#
# Copy an image file of your own in this folder,
# open it and display it.

# Write your code here

img  = plt.imread('komodo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img,cmap='gray')

savepic("gray")


# Task #2: Create your Own Kernel.
#
# In this cell, try your hand at coming up with a kernel with some logic
# in its values, and try to predict how it will behave.
# Your kernel's dimensions may be 3x3, 5x5 or 7x7.


# Modify this kernel definition
kernel = np.matrix([[2,  1,  0],
                    [1,  0, -1],
                    [0, -1, -2]])




# Run this cell to test your kernel
# Don't modify this cell, just run it

filtered1 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
rcParams['figure.figsize'] = 20,8

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img,cmap='gray')
ax[1].imshow(filtered1,cmap='gray')


savepic("sober-filter")


# Task #3: Create a second kernel
#
# Now try to come up with a second kernel.
# You may want to perform something related to the first kernel,
# or maybe something totally different.
#
# Here's the same code as before for you to modify the kernel.


# Modify this kernel definition
kernel = np.matrix([[0, -1, -2],
                    [1,  0, -1],
                    [2,  1,  0]])

filtered2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
rcParams['figure.figsize'] = 20,8

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img,cmap='gray')
ax[1].imshow(filtered2,cmap='gray')

savepic("sober-filter-2")


# Now let's see the two images side by side

fig, ax = plt.subplots(1,2)
ax[0].imshow(filtered1,cmap='gray')
ax[1].imshow(filtered2,cmap='gray')


savepic("comparison")