index = 0

def savepic(label):
    global index
    plt.savefig(f"07-05-{index}-{label}.png")
    index = index + 1
    plt.close()


# Don't modify this cell, just run it.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams['figure.figsize'] = (17, 14)




# In this cell, we have the picture taken by the camera in the ceiling,
# as requested by the robot.
#
# Run this cell and move on to the next one.

img = plt.imread('warehouse.bmp')
plt.axis("off")
plt.imshow(img,cmap='gray')

savepic("warehouse")





# In this cell we have a special operation to create a color mask.
#
# This procedure is described in OpenCV's tutorials website:
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

# Convert RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Define range of color in HSV
lower = np.array([10,50,50])
upper = np.array([90,255,255])

# Threshold the HSV image to get only brown colors
mask = cv2.inRange(hsv, lower, upper)

plt.imshow(mask,cmap='gray')

savepic("mask")


# Run this cell to appreciate the lingering white dots

rcParams['figure.figsize'] = (15, 14)
plt.imshow(mask[:300,300:800],cmap='gray')

savepic("zoom-in")


# Task #1: Get rid of the noise
# In this cell, use morphological transformations to get rid of the
# white dots throughout the mask.

rcParams['figure.figsize'] = (17, 14)
# Write your code here

kernel3 = np.ones((3,3),np.uint8)
mask = cv2.erode(mask,kernel3,iterations = 1)
plt.imshow(mask,cmap='gray')

savepic("erode-3x3")



# Task #2: Make the blobs grow.
# In this cell, use morphological transformations to make the obstacle
# blobs grow.
# It's simple: Just dilate the mask with a 5x5 kernel about 10 times
# or with a 3x3 kernel about 20 times.

# Write your code here

kernel5 = np.ones((5,5),np.uint8)

mask = cv2.dilate(mask,kernel5,iterations = 10)
plt.imshow(mask,cmap='gray')


savepic("dilate-5x5-10")

