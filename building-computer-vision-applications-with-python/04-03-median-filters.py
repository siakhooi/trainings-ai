index = 0

def savepic(label):
    global index
    plt.savefig(f"04-03-{index}-{label}.png")
    index = index + 1
    plt.close()

# Let's work with a grainy picture

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
from EC_CV import *
import cv2

rcParams['figure.figsize'] = (24, 10)

img = plt.imread('field.jpg')
plt.axis("off")
plt.imshow(img)

savepic("original")

# OpenCV's functions work with color and grayscale images

median = cv2.medianBlur(img,5)
plt.axis("off")
plt.imshow(median)

savepic("median")

# Let's see an average blur side by side with a median blur

rcParams['figure.figsize'] = (20,8)

kernel = np.ones((5, 5), np.float32) / 25
average = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(average)
ax[1].imshow(median)

savepic("comparison")

# Let's zoom in to the leftmost section

rcParams['figure.figsize'] = (20, 20)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,:400])
ax[1].imshow(median[:,:400])

savepic("comparison")

# Now a middle section

fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,400:800])
ax[1].imshow(median[:,400:800])

savepic("comparison")

# Lastly, the rightmost section

fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,800:1200])
ax[1].imshow(median[:,800:1200])

savepic("comparison")

# Now let's see it for a different picture

rcParams['figure.figsize'] = (24, 10)

img = plt.imread('workers.jpg')
plt.axis("off")
plt.imshow(img)

savepic("original")

# OpenCV's functions work with color and grayscale images

median = cv2.medianBlur(img,3)
plt.axis("off")
plt.imshow(median)

savepic("median")

# Let's see an average blur side by side with a median blur

rcParams['figure.figsize'] = (20,8)

kernel = np.ones((3, 3), np.float32) / 9
average = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(average)
ax[1].imshow(median)

savepic("comparison")

# Let's zoom in to the leftmost section

rcParams['figure.figsize'] = (20, 20)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,:400])
ax[1].imshow(median[:,:400])

savepic("comparison")

# Now a middle section

fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,400:800])
ax[1].imshow(median[:,400:800])

savepic("comparison")


# Lastly, the rightmost section

fig, ax = plt.subplots(1,2)
ax[0].imshow(average[:,700:1100])
ax[1].imshow(median[:,700:1100])

savepic("comparison")


# Now you try it with this picture of a face.
# Experiment with several kernel sizes.

rcParams['figure.figsize'] = (24, 10)

img = plt.imread('face.jpg')
plt.axis("off")
plt.imshow(img)

savepic("original")

