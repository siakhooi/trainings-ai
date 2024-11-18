index = 0

def savepic(label):
    global index
    plt.savefig(f"05-06-{index}-{label}.png")
    index = index + 1
    plt.close()


# Don't modify this cell, just run it.
# Both functions are included in EC_CV:
#                          downscale()
#                          upscale_by_2()

import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams['figure.figsize'] = (20,8)



# Task #1: Print an image
#
# Copy an image file of your own in this folder,
# open it and display it.

# Write your code here

original  = plt.imread('pizza.bmp')
plt.imshow(original)
print(np.shape(original))

savepic("original")

# Task #2: Downscale that image.
#
# In this cell, downscale your image to the resolution of your choice.
# Feel free to use the downscale() function, or any other function
# either made by yourself, or from a library, like cv2.resize()

# Write your code here

smaller = adapt_image(downscale(original,4))
smaller_OCV = cv2.resize(original,(389,238))
print(np.shape(smaller))

# Let's see the two images side by side
fig, ax = plt.subplots(1,2)
ax[0].imshow(smaller)
ax[1].imshow(smaller_OCV)


savepic("smaller")


# Task #3: Upscale that image.
#
# In this cell, upscale your image to the resolution of your choice.
# Feel free to use the upscale_by_2() function, or any other function
# either made by yourself, or from a library, like cv2.resize()
#

# Write your code here

larger = upscale_by_2(smaller)
larger = upscale_by_2(larger)
larger = adapt_image(larger)
print(np.shape(larger))

larger_OCV = cv2.resize(smaller,(1556,952))

# Let's see the two images side by side
fig, ax = plt.subplots(1,2)
ax[0].imshow(larger)
ax[1].imshow(larger_OCV)

savepic("larger")
