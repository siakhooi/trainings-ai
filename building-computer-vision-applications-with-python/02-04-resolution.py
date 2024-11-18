index = 0

def savepic(label):
    global index
    plt.savefig(f"02-04-{index}-{label}.png")
    index = index + 1


# Let's open 3 JPEG image files with different resolutions

import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

plt.rcParams['figure.figsize'] = (10, 8) # (Width, Height) supposedly in inches

img1 = plt.imread('dog800x600.jpg')
plt.imshow(img1)
savepic("dog800x600")

img2 = plt.imread('dog300x225.jpg')
plt.imshow(img2)
savepic("dog300x225")

img3 = plt.imread('dog120x90.jpg')
plt.imshow(img3)
savepic("dog120x90")

plt.rcParams['figure.figsize'] = (1, 1) # (Width, Height) supposedly in inches
plt.imshow(img3)
