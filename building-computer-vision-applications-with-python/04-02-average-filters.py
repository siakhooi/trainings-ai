index = 0

def savepic(label):
    global index
    plt.savefig(f"04-02-{index}-{label}.png")
    index = index + 1
    plt.close()



# Let's work with the picture of a house

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
from EC_CV import *
import cv2

rcParams['figure.figsize'] = (20,8)

img = plt.imread('house.jpg')
plt.imshow(img)
print(np.shape(img))
savepic("original")


# OpenCV's filter2D function works with color and grayscale images

kernel = np.ones((7, 7), np.float32) / 49
blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
plt.imshow(blurred)
savepic("blurred")


# Let's compare the images side by side

rcParams['figure.figsize'] = (20,8)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(blurred)

savepic("comparison")

rcParams['figure.figsize'] = (20, 20)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img[:,400:800])
ax[1].imshow(blurred[:,400:800])


# Now let's see the same thing with a 3x3 kernel

rcParams['figure.figsize'] = (20, 20)

kernel = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img[:,400:800])
ax[1].imshow(blurred[:,400:800])

