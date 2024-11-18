index = 0

def savepic(label):
    global index
    plt.savefig(f"03-03-{index}-{label}.png")
    index = index + 1
    plt.close()


# Let's work with the play space picture

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
from EC_CV import *

rcParams['figure.figsize'] = 20,8

toys = adapt_PNG(plt.imread('playspace.png'))
toys_wgt = np.dot(toys[...,:3], [0.299,0.587,0.114])
plt.axis("off")
plt.imshow(toys_wgt,cmap='gray')
savepic("original")

# Let's turn each pixel to total black or total white.
# We'll use the grayscale_to_BW() function from EC_CV

rcParams['figure.figsize'] = 20,8

toys_BW = grayscale_to_BW(toys_wgt,127)
plt.imshow(toys_BW, cmap = 'gray')
savepic("BW")


# Let's see a histogram of the grayscale image

rcParams['figure.figsize'] = (14,8)
plt.hist(toys_wgt.ravel(),256,[0,255])
plt.title('Histogram')
plt.xticks(np.arange(0, 255, 10))
plt.show()

savepic("histogram")


# Now let's create a new Black and White image with a threshold of 115
rcParams['figure.figsize'] = (20,8)

toys_BW2 = grayscale_to_BW(toys_wgt,115)

fig, ax = plt.subplots(1,2)
ax[0].imshow(toys_BW, cmap='gray')
ax[1].imshow(toys_BW2, cmap='gray')

savepic("BW-115")
