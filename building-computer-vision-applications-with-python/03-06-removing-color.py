index = 0

def savepic(label):
    global index
    plt.savefig(f"03-06-{index}-{label}.png")
    index = index + 1
    plt.close()


# Don't modify this cell, just run it.

import numpy as np
import matplotlib.pyplot as plt
from EC_CV import *
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
import cv2

rcParams['figure.figsize'] = (10, 8)


# Task #1: Print a color image
#
# Copy an image file of your own in this folder,
# open it and display it.

# Write your code here

img_c  = plt.imread('sunset.jpg')
plt.imshow(img_c)

savepic("original")



# Task #2: Convert the color image to grayscale.
#
# Use the weights we used before:
#                              Red: 0.299
#                            Green: 0.587
#                             Blue: 0.114
#

# Write your code here

img_gs = np.dot(img_c[...,:3], [0.299,0.587,0.114])
plt.imshow(img_gs, cmap='gray')

savepic("grayscale")


# Task #3: Plot a histogram
#
# In this cell, plot a histogram of your grayscale image
# and try to come up with a suitable global threshold.

# Write your code here
rcParams['figure.figsize'] = 14,8
plt.hist(img_gs.ravel(),256,[0,255])
plt.title('Histogram')
plt.xticks(np.arange(0, 255, 10))
plt.show()



savepic("histogram")

# Task #4: Compare two Black and White Images
#
# Pick two global thresholds from the histogram above and compare
# the two resulting images. Feel free to use OpenCV.

# Write your code here

img_BW1 = grayscale_to_BW(img_gs,115)
img_BW2 = grayscale_to_BW(img_gs,175)

# figure size in inches
rcParams['figure.figsize'] = 18,8

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_BW1, cmap='gray')
ax[1].imshow(img_BW2, cmap='gray')


savepic("compares")
