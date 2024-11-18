index = 0


def savepic(label):
    global index
    plt.savefig(f"02-06-{index}-{label}.png")
    index = index + 1


# Run this cell to open 3 images.
# Don't modify this cell.

import numpy as np
import matplotlib.pyplot as plt
from EC_CV import *

# %config InlineBackend.figure_formats = 'retina'

# We'll work with 1 JPEG and 2 PNGs
traffic = plt.imread("traffic.jpg")
penguins = plt.imread("penguins.png")
icepops = plt.imread("icepops.png")


# Task #1: Find the RGBA file
#
# One of these images is a JPEG file, and the other two are PNGs.
# One of the PNGs is encoded as RGB, and the other as RGBA.
# Identify the PNG file that's encoded as RGBA.
# You may do this any way you want: Get the dimensions, print a pixel, etc.

# Write your code here

print(np.shape(traffic))
print(np.shape(penguins))
print(np.shape(icepops))


# Task #2: Convert the images to an 8-bit unsigned integer RGB encoding
#
# Depending on each image, the type of its values may be a floating-point
# number or an integer.
# Since we want to work with 8-bit RGB, we need to make sure all our
# arrays are encoded that way.
# Feel free to use the functions in the EC_CV.py source file (imported above).

# Write your code here

penguins = adapt_PNG(penguins)
icepops = adapt_image(icepops * 255)  # Could've used adapt_PNG()

print(np.shape(traffic))
print(type(traffic[0, 0, 0]))
print(np.shape(penguins))
print(type(penguins[0, 0, 0]))
print(np.shape(icepops))
print(type(icepops[0, 0, 0]))


# Task #3 (3 cells): Print the images
#
# In this cell, display traffic, showing its axes

# Write your code here

plt.imshow(traffic)

savepic("traffic")

# In this cell, display penguins, hiding its axes

# Write your code here

plt.axis("off")
plt.imshow(penguins)

savepic("penguins")

# In this cell, display icepops, hiding its axes

# Write your code here

plt.axis("off")
plt.imshow(icepops)

savepic("icepops")


# Task #4 (3 cells): Manipulate the images
#
# In this cell, flip traffic horizontally (mirror image)
# and display the resulting image

# Write your code here

traffic = np.fliplr(traffic)
plt.axis("off")
plt.imshow(traffic)

savepic("traffic-flipped")


# In this cell, rotate penguins 90 degrees clockwise
# and display the resulting image.
#
# This rotation means -90 degrees, or +270 degrees.

# Write your code here

penguins = np.rot90(penguins, 3)
plt.axis("off")
plt.imshow(penguins)

savepic("penguins-rotated")


# Lastly, let's multiply icepops by 2
#
# Don't modify this cell, just run it.
# Then, run the next cell 8 times.

# If you lose count, or you need to run the experiment again,
# you may rerun this cell to reset img to icepops.

img = icepops


# Run this cell 10 times and see what happens each time.
# Don't modify this cell, just run it.

img *= 2
img *= 2
img *= 2
img *= 2
img *= 2
img *= 2
img *= 2
img *= 2
plt.axis("off")
plt.imshow(img)

savepic("icepops-multiplied")

# *2 equivable to shift bit to left by 1, 8 times shifted all 8 bits, thus left 0
