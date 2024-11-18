index = 0

def savepic(label):
    global index
    plt.savefig(f"02-05-{index}-{label}.png")
    index = index + 1

# Let's open a picture and display it without the axes

import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = 'retina'

img = plt.imread('street.jpg')
plt.axis("off")
plt.imshow(img)

savepic("original")



# Now let's rotate it 90 degrees counter clockwise

img = np.rot90(img,1)
plt.axis("off")
plt.imshow(img)
savepic("rot90")


# Now let's rotate it 90 degrees 3 times at once

img = np.rot90(img,3)
plt.axis("off")
plt.imshow(img)
savepic("rot90x3")



# Let's perform a horizontal flip (invert left and right).
# This is how you would see the image reflected in a mirror.

img = np.fliplr(img)
plt.axis("off")
plt.imshow(img)

savepic("fliplr")


# Now let's try a vertical flip (invert up and down)

img = np.flipud(img)
plt.axis("off")
plt.imshow(img)

savepic("flipud")
