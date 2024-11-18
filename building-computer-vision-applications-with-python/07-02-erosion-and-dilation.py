index = 0

def savepic(label):
    global index
    plt.savefig(f"07-02-{index}-{label}.png")
    index = index + 1
    plt.close()

# Erosion and Dilation
# Erosion and dilation work with a kernel that slides through an image, as in convolution.
# - For erosion, a pixel is turned black if there are black pixels in its neighborhood area.
# - For dilation, a pixel is turned white if there are white pixels in its neighborhood area.

# Let's get to know the cv2.erode() and cv2.dilate() functions.



import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams['figure.figsize'] = (20,8)




# Let's open a black and white picture

img = plt.imread('hi_there.bmp')
img = np.dot(img[...,:3], [0.299,0.587,0.114])
print(np.shape(img))
plt.imshow(img,cmap='gray')

savepic("hi-there")


# Now let's perform a 3x3 erosion

kernel3 = np.ones((3,3),np.uint8)
img2 = img
img2 = cv2.erode(img2,kernel3,iterations = 1)
plt.imshow(img2,cmap='gray')
img3 = img2

savepic("erosion-3x3")


# Now let's dilate it with a 5x5 kernel

kernel5 = np.ones((5,5),np.uint8)
img2 = cv2.dilate(img2,kernel5,iterations = 1)
plt.imshow(img2,cmap='gray')

savepic("dilation-5x5")


# Now let's erode the dilated image

img2 = cv2.erode(img2,kernel5,iterations = 1)
plt.imshow(img2,cmap='gray')

savepic("erosion-5x5")


# Let's see the image before dilating and eroding, and after

fig, ax = plt.subplots(1,2)
ax[0].imshow(img3,cmap='gray')
ax[1].imshow(img2,cmap='gray')

savepic("before-after")


# Let's open a different black and white picture

img = plt.imread('shapes.bmp')
print(np.shape(img))
plt.imshow(img,cmap='gray')

savepic("shapes")



# Let's erode those shapes 4 times with a 3x3 kernel

img2 = img
img2 = cv2.erode(img2,kernel3,iterations = 4)
plt.imshow(img2,cmap='gray')

savepic("erode-4x3")

# Now let's dilate those shapes twice with a 5x5 kernel

img2 = cv2.dilate(img2, kernel5, iterations = 2)
plt.imshow(img2,cmap='gray')

savepic("dilate-2x5")


# Now let's erode those shapes 8 times with a 5x5 kernel

img3 = img2
img2 = cv2.erode(img2, kernel5, iterations = 8)
plt.imshow(img2,cmap='gray')

savepic("erode-8x5")


# Now let's dilate those shapes 8 times with a 5x5 kernel

img2 = cv2.dilate(img2, kernel5, iterations = 8)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img3,cmap='gray')
ax[1].imshow(img2,cmap='gray')

savepic("before-after-8x5")

