# Let's create a simple 3x3 Grayscale image.
# Each pixel is a number (0-255) representing intensity.

import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = 'retina'

index=0
def savepic(label):
    global index
    plt.savefig(f'02-01-{index}-{label}.png')
    index=index+1


img = np.array([[   0, 255,   0],   #   black,  white,     black
                [  50, 200,  50],   #    dark,  light,      dark
                [ 110, 127, 140]])  # mid-dark,   mid, mid-light

plt.imshow(img, cmap='gray')
savepic("int")

# This is how it looks in a text representation

print(img)
type(img[0,0])


# Now let's create a simple 3x3 RGB image.
# This time each pixel is an [R,G,B] triad.

img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],   #   red,   green,     blue
                [[  0, 255, 255], [255,   0, 255], [255, 255,   0]],   #  cyan, magenta,   yellow
                [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])  # black,   white, gray 50%
plt.axis("off")
plt.imshow(img)
savepic("array-3-int")


# This is how it looks in a text representation

print(img)
type(img[0,0,0])



# Now let's create the same image with floating point numbers.

img = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],    #  red,   green,     blue
                [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],    # cyan, magenta,   yellow
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])  # black,   white, gray 50%
plt.imshow(img)
savepic("array-3-float")



# This is how it looks in a text representation

print(img)
type(img[0,0,0])