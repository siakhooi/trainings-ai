index = 0

def savepic(label):
    global index
    plt.savefig(f"06-02-{index}-{label}.png")
    index = index + 1
    plt.close()


# Let's work with the picture of a green wall.

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
from EC_CV import *
import cv2

rcParams['figure.figsize'] = (20,28)

img = plt.imread('TheWall.bmp')
plt.imshow(img)
np.shape(img)

savepic("original")

# Now let's split the image in two separate parts, right at the piece of litter.

imgL = img[:,:870]
imgR = img[:,870:]
print(np.shape(imgL))
print(np.shape(imgR))

fig, ax = plt.subplots(1,2)
ax[0].imshow(imgL)
ax[1].imshow(imgR)

savepic("split")


# Let's look at the two images put together in a vertical
# straight line at the middle of their overlap

overlap = 70 # also try 52 and 46
heightL, widthL, temp = imgL.shape
heightR, widthR, temp = imgR.shape

stitch = np.concatenate((imgL[:,:widthL-int(overlap/2)], imgR[:,int(overlap/2):]), axis=1)
plt.imshow(stitch)

savepic("stitch")

# Let's zoom in to appreciate the splitting artifact

rcParams['figure.figsize'] = (20,10)
tiny_stitch = stitch[int(heightL/2)-overlap*2:int(heightL/2)+overlap*2,widthL-overlap*2:widthL+overlap]
plt.imshow(tiny_stitch)

savepic("zoomed")


## Let's create a seam at a very small part of the overlap first

# Using a section of 100 vertical pixels, let's look at
# the Left and Right components of the overlap.

rcParams['figure.figsize'] = (20,8)

tinyL = imgL[:100,widthL-overlap:]
tinyR = imgR[:100,:overlap]

# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(tinyL)
ax[1].imshow(tinyR)

savepic("tiny")

# Now let's caluclate their difference.

rcParams['figure.figsize'] = (20,10)

# Turn both sections to grayscale
tinyL_g = np.dot(tinyL[...,:3], [0.299,0.587,0.114])
tinyR_g = np.dot(tinyR[...,:3], [0.299,0.587,0.114])

# Calculate their squared difference
diff = cv2.subtract(tinyL_g,tinyR_g)
diff = cv2.multiply(diff,diff)

# Display the squared difference
plt.imshow(diff,cmap='gray')
np.shape(diff)

savepic("diff")


# Let's calculate the seam for the small squared difference
# The get_seam() function is defined in EC_CV.py

height, width = diff.shape
my_seam = get_seam(diff)

# Show the seam in the squared difference with white pixels
diff_seam = diff.copy()
height, width = diff.shape
for i in range(height):
    diff_seam[i,int(my_seam[i])] = 255 * 255 # This is white squared
plt.imshow(diff_seam,cmap='gray')
print(height)
print(width)

savepic("diff_seam")


## Now let's create the whole seam


# Using the whole overlap, let's look at its Left and Right components.

rcParams['figure.figsize'] = (20,8)
tinyL = imgL[:,widthL-overlap:]
tinyR = imgR[:,:overlap]

fig, ax = plt.subplots(1,2)
ax[0].imshow(tinyL)
ax[1].imshow(tinyR)

savepic("whole")


# Now let's caluclate their difference.

# Turn both sections to grayscale
tinyL_g = np.dot(tinyL[...,:3], [0.299,0.587,0.114])
tinyR_g = np.dot(tinyR[...,:3], [0.299,0.587,0.114])

# Calculate their squared difference
diff = cv2.subtract(tinyL_g,tinyR_g)
diff = cv2.multiply(diff,diff)

# Display the squared difference
plt.imshow(diff,cmap='gray')
np.shape(diff)


savepic("diff")


# Let's calculate the seam for the squared difference
rcParams['figure.figsize'] = (20,28)
height, width = diff.shape
my_seam = get_seam(diff)

# Show the seam in the squared difference with white pixels
diff_seam = diff.copy()
height, width = diff.shape
for i in range(height):
    diff_seam[i,int(my_seam[i])] = 255 * 255 # This is white squared
plt.imshow(diff_seam,cmap='gray')
print(height)
print(width)

savepic ("diff_seam")


## Stitching the sections together


# First let's create the middle section.
# This is the overlapping section, with pixels from both Left and
# Right components. The seam determines where Left pixels end
# and Right pixels start.

middle = tinyL.copy()    # Start with Left overlap.

# For every row in the middle section, replace Left pixels with
# Right pixels starting at the seam.
for i in range(height):
    j = int(my_seam[i])
    while j < width:
        middle[i,j] = tinyR[i,j]
        j += 1

# Now let's create a marked middle section to show the seam with red pixels.
middle_marked = middle.copy()
for i in range(height):
    middle_marked[i,int(my_seam[i])] = np.array((255,0,0))

rcParams['figure.figsize'] = (20,28)
plt.imshow(middle_marked,cmap='gray')

savepic("middle_marked")


# Now let's create the stitched image consisting of:
#    The Leftmost Image (minus the overlap)
#       + The Middle Section with the seam in red
#          + The Rightmost Section (minus the overlap)

cut_stitch = np.concatenate((imgL[:,:widthL-overlap], middle_marked), axis=1)
cut_stitch = np.concatenate((cut_stitch, imgR[:,overlap:]), axis=1)
plt.imshow(cut_stitch)

savepic("cut_stitch")


# Now let's create the stitched image, not showing the seam.

cut_stitch = np.concatenate((imgL[:,:widthL-overlap], middle), axis=1)
cut_stitch = np.concatenate((cut_stitch, imgR[:,overlap:]), axis=1)
plt.imshow(cut_stitch)

savepic("cut_stitch")


## Lastly let's zoom in to see the difference


# Display both zoomed-in results

rcParams['figure.figsize'] = (20,10)

tiny_stitch = stitch[int(heightL/2)-overlap*2:int(heightL/2)+overlap*2,widthL-overlap*2:widthL+overlap]
tiny_cut_stitch = cut_stitch[int(heightL/2)-overlap*2:int(heightL/2)+overlap*2,widthL-overlap*2:widthL+overlap]

fig, ax = plt.subplots(1,2)
ax[0].imshow(tiny_stitch)
ax[1].imshow(tiny_cut_stitch)


savepic("zoomed")


# Finally let's save the stiched image into a file
plt.imsave("stitched.bmp",cut_stitch)

savepic("final")

# # Just for fun
# Let's invert the two images and do the whole thing again!


# Here we have the inverted sections.
imgR = img[:,:870]
imgL = img[:,870:]

overlap = 70
heightL, widthL, temp = imgL.shape
heightR, widthR, temp = imgR.shape

stitch = np.concatenate((imgL[:,:widthL-int(overlap/2)], imgR[:,int(overlap/2):]), axis=1)

# Left and righe overlap components.
tinyL = imgL[:,widthL-overlap:]
tinyR = imgR[:,:overlap]

# Now let's caluclate their difference.
# Turn both sections to grayscale
tinyL_g = np.dot(tinyL[...,:3], [0.299,0.587,0.114])
tinyR_g = np.dot(tinyR[...,:3], [0.299,0.587,0.114])

# Calculate their squared difference
diff = cv2.subtract(tinyL_g,tinyR_g)
diff = cv2.multiply(diff,diff)

# Let's calculate the seam for the squared difference
height, width = diff.shape
my_seam = get_seam(diff)


# First let's create the middle section.
middle = tinyL.copy()    # Start with Left overlap.
# For every row in the middle section, replace Left pixels with
# Right pixels starting at the seam.
for i in range(height):
    j = int(my_seam[i])
    while j < width:
        middle[i,j] = tinyR[i,j]
        j += 1

# Now let's create the stitched image.
cut_stitch = np.concatenate((imgL[:,:widthL-overlap], middle), axis=1)
cut_stitch = np.concatenate((cut_stitch, imgR[:,overlap:]), axis=1)
plt.imshow(cut_stitch)


savepic("cut_stitch_inverted")


# Display both zoomed-in results

rcParams['figure.figsize'] = (20,10)

tiny_stitch = stitch[int(heightL/2)-overlap*2:int(heightL/2)+overlap*2,widthL-overlap*2:widthL+overlap]
tiny_cut_stitch = cut_stitch[int(heightL/2)-overlap*2:int(heightL/2)+overlap*2,widthL-overlap*2:widthL+overlap]

fig, ax = plt.subplots(1,2)
ax[0].imshow(tiny_stitch)
ax[1].imshow(tiny_cut_stitch)

savepic("zoomed_inverted")


# Let's look at the grass
tiny_stitch = stitch[750:,widthL-overlap*2:widthL+overlap]
tiny_cut_stitch = cut_stitch[750:,widthL-overlap*2:widthL+overlap]

fig, ax = plt.subplots(1,2)
ax[0].imshow(tiny_stitch)
ax[1].imshow(tiny_cut_stitch)


savepic("zoomed_inverted_grass")
