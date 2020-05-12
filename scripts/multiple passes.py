import base64
import io
import pickle
import math
from imageio import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt

b64Img = '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'

data = base64.b64decode(b64Img)
ionicImg = cv2.imread('persona.png', cv2.IMREAD_UNCHANGED)


borderType = cv2.BORDER_CONSTANT
#img = cv2.imread(ionicImg, cv2.IMREAD_UNCHANGED)
#make mask of where the transparent bits are
trans_mask = ionicImg[:, :, 3] == 0
#replace areas of transparency with white and not transparent
ionicImg[trans_mask] = [255, 255, 255, 255]
gray = cv2.bitwise_not(cv2.cvtColor(ionicImg, cv2.COLOR_BGRA2GRAY))
a = np.sum(gray, axis=0) > 0
b = np.sum(gray, axis=1) > 0
col_sum = np.where(a)
row_sum = np.where(b)
y1, y2 = row_sum[0][0], math.ceil(row_sum[0][-1])
x1, x2 = math.floor(col_sum[0][0]), col_sum[0][-1]

""" cropped_image = gray[y1:y2, x1:x2]
# Initialize arguments for the filter
top = int(0.1 * cropped_image.shape[0])  # shape[0] = rows
bottom = top
left = int(0.1 * cropped_image.shape[1])  # shape[1] = cols
right = left
value = 0
paddedImg = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, borderType, None, value)
paddedImg = paddedImg/255


imgRe = cv2.resize(paddedImg, (60, 60), interpolation=cv2.INTER_AREA) """




plt.imshow(gray, cmap='binary')

""" plt.subplot(212)
plt.imshow(imgRe, cmap='binary')"""
plt.show()


