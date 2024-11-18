index = 0

def savepic(label):
    global index
    plt.savefig(f"05-04-{index}-{label}.png")
    index = index + 1
    plt.close()

# Let's start with a low-resolution picture

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
from EC_CV import *
import cv2

rcParams['figure.figsize'] = (20,8)

img = plt.imread('smaller.bmp')
plt.imshow(img)
np.shape(img)

savepic("smaller")


# Let's increase the resolution by a linear factor of 2 (4 times larger in area)
# The upscale_by_2() function is defined in EC_CV.py

img_l = upscale_by_2(img)
img_l = adapt_image(img_l)
plt.imshow(img_l)

savepic("larger")

# Let's see the two images side by side

fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img_l)


savepic("both")

# Let's perform a second enlargement

img_l = upscale_by_2(img_l)
img_l = adapt_image(img_l)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img_l)

savepic("both_larger")



# Let's save the larger image into a file

#plt.imsave("larger.bmp",img_l)



# Now that we've reduced an image to 1/16 its size
# and stretched it back up 16 times,
# let's compare the original and final images.

volcano = plt.imread('volcano.jpg')

print(np.shape(volcano))
print(np.shape(img_l))

fig, ax = plt.subplots(1,2)
ax[0].imshow(volcano)
ax[1].imshow(img_l)


savepic("volcano_both")
