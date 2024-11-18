index = 0

def savepic(label):
    global index
    plt.savefig(f"05-02-{index}-{label}.png")
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

img = plt.imread('volcano.jpg')
plt.imshow(img)
np.shape(img)

savepic("volcano")

# Let's reduce the resolution by a linear factor of 4 (16 times smaller in area)
# The downscale() function is defined in EC_CV.py

img_s = downscale(img,4)
img_s = adapt_image(img_s)
plt.imshow(img_s)
print(np.shape(img_s))

savepic("volcano_small")

# Let's see the two images side by side

print(np.shape(img))
print(np.shape(img_s))

fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img_s)

savepic("volcano_both")


# Now let's do the same at the smaller image's size

rcParams['figure.figsize'] = (3.5,8)

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img_s)

savepic("volcano_both_small")



# Now let's save the smaller image into a file

#plt.imsave("smaller.bmp",img_s)




# Let's repeat the process for another image

rcParams['figure.figsize'] = (20,8)

img = plt.imread('meat.bmp')
img_s = adapt_image(downscale(img,10))

print(np.shape(img))
print(np.shape(img_s))

fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img_s)


savepic("meat_both")
