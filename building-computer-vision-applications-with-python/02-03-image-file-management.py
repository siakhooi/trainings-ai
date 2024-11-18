index = 0

def savepic(label):
    global index
    plt.savefig(f"02-03-{index}-{label}.png")
    index = index + 1

# Let's open a JPEG image file using the pyplot.imread function.

import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = 'retina'

jpeg = plt.imread('stickanimals.jpg')
plt.imshow(jpeg)
savepic("jpeg")

# Now let's open an equivalent PNG image file

png = plt.imread('stickanimalsRGBA.png')
plt.imshow(png)
savepic("png")


# Let's see their dimensions

print(np.shape(jpeg))
print(np.shape(png))



# Let's see their upper left pixel

print(jpeg[0,0])
print(png[0,0])




# Now let's see their data type

print('JPEG image type: ', type(jpeg[0,0,0]))
print('PNG image type: ', type(png[0,0,0]))



# So from now on, when opening PNG files for these exercises,
# you may use the adapt_PNG function from EC_CV like this:

from EC_CV import *

img = adapt_PNG(plt.imread("stickanimalsRGBA.png"))

print('PNG image data type: ', type(img[0,0,0]))
print(img[0,0])
plt.imshow(img)


savepic("adapted-png")

# Lastly, let's save a file.
# This time let's extract the horse as a subarray.

horsie = img[250:600,200:400,:]
plt.imsave('horsie.jpg',horsie)
