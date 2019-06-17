#%% [markdown]
# ### Loading required modules

import cv2
import numpy as np
from scipy import misc, signal
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt

i = misc.ascent()

#%% [markdown]
# ### Drawing the sample image using matplotlib

plt.grid(False)
# plt.axis('off')
plt.imshow(i)
plt.show()

#%% [markdown]
# ### Create a 3x3 convolution filter

filter_1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter_2 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filter_3 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
weight = 1

#%% [markdown]
# ### Applying the convolution with different filters on image

i_transformed = signal.convolve2d(i, filter_3, mode="same")
i_transformed[i_transformed < 0] = 0
i_transformed[i_transformed > 255] = 255

#%% [markdown]
# ### Plot the result of applying filter on image

plt.grid(False)
plt.imshow(i_transformed, cmap="gray")
plt.show()

#%% [markdown]
# ### Applying a 2x2 max pooling filter on the image

i_maxpooled = maximum_filter(i_transformed, footprint=np.ones((2, 2)))

#%% [markdown]
# ### Plot the result of max pooling

plt.grid(False)
plt.imshow(i_maxpooled)
plt.show()
