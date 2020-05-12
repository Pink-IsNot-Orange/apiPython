import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
borderType = cv2.BORDER_CONSTANT
windowName = "copyMakeBorder Demo"

dim = (64,128)

imagen = cv2.imread('seis.png', cv2.IMREAD_UNCHANGED)
#make mask of where the transparent bits are
trans_mask = imagen[:, :, 3] == 0
#replace areas of transparency with white and not transparent
imagen[trans_mask] = [255, 255, 255, 255]
gray = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)

col_sum = np.where(np.sum(gray, axis=0) > 0)
row_sum = np.where(np.sum(gray, axis=1) > 0)
y1, y2 = row_sum[0][0], math.ceil(row_sum[0][-1])
x1, x2 = math.floor(col_sum[0][0]), col_sum[0][-1]
cropped_image = gray[y1:y2, x1:x2]

paddedImg = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
#creating hog features 
fd, hog_image = hog(paddedImg, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(paddedImg, cmap='gray') 
ax1.set_title('Input image') 

ax2.imshow(hog_image, cmap='gray') 
ax2.set_title('Histogram of Oriented Gradients')

# Rescale histogram for better display 
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

ax3.imshow(hog_image_rescaled, cmap='gray') 
ax3.set_title('HOG Exposure rescaled')

plt.show()

## SVM Implementation comming
