import glob
import cv2
import numpy as np
import math
import functools
import matplotlib.pyplot as plt

#FILES 
persona = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\persona\*.png')
uno = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\1\*.png')
dos = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\2\*.png')
cuatro = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\4\*.png')
cinco = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\5\*.png')
seis = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\6\*.png')
siete = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\7\*.png')
ocho = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\8\*.png')
nueve = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\9\*.png')
diez = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\10\*.png')
mil = glob.glob(r'C:\Users\JONA-PC\Desktop\pythonApi\data\1000\*.png')

#INICIALIZACIONES
persona_train = []
uno_train = []
dos_train = []
cuatro_train = []
cinco_train = []
seis_train = []
siete_train = []
ocho_train = []
nueve_train = []
diez_train = []
mil_train = []

persona_label = np.empty(1000)
uno_label = np.empty(1000)
dos_label = np.empty(1000)
cuatro_label = np.empty(1000)
cinco_label = np.empty(1000)
seis_label = np.empty(1000)
siete_label = np.empty(1000)
ocho_label = np.empty(1000)
nueve_label = np.empty(1000)
diez_label = np.empty(1000)
mil_label = np.empty(1000)

persona_label[:] = 0.
uno_label[:] = 1.
dos_label[:] = 2.
cuatro_label[:] = 3.
cinco_label[:] = 4.
seis_label[:] = 5.
siete_label[:] = 6.
ocho_label[:] = 7.
nueve_label[:] = 8.
diez_label[:] = 9.
mil_label[:] = 10.


img = [persona, uno, dos, cuatro, cinco, seis, siete, ocho, nueve, diez, mil]
size = functools.reduce(lambda count, element: count + len(element), img, 0)

kanjis = [persona_train, uno_train, dos_train, cuatro_train, cinco_train, seis_train, siete_train, ocho_train, nueve_train, diez_train, mil_train]
labels = [persona_label, uno_label, dos_label, cuatro_label, cinco_label, seis_label, siete_label, ocho_label, nueve_label, diez_label, mil_label]

dim = (150, 150)

train = np.empty((size, dim[0]*dim[1]), dtype='float32')
train_labels = np.empty(size, dtype='int64')

borderType = cv2.BORDER_CONSTANT
windowName = "copyMakeBorder Demo"
#RESIZE MADE HERE
for i in range(len(img)):
    for myFiles in img[i]:
        imagen = cv2.imread(myFiles, cv2.IMREAD_UNCHANGED)
        #make mask of where the transparent bits are
        trans_mask = imagen[:, :, 3] == 0
        #replace areas of transparency with white and not transparent
        imagen[trans_mask] = [255, 255, 255, 255]
        gray = cv2.bitwise_not(cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY))

        col_sum = np.where(np.sum(gray, axis=0) > 0)
        row_sum = np.where(np.sum(gray, axis=1) > 0)
        y1, y2 = row_sum[0][0], math.ceil(row_sum[0][-1])
        x1, x2 = math.floor(col_sum[0][0]), col_sum[0][-1]
        cropped_image = gray[y1:y2, x1:x2]
        # Initialize arguments for the filter
        top = int(0.1 * cropped_image.shape[0])  # shape[0] = rows
        bottom = top
        left = int(0.1 * cropped_image.shape[1])  # shape[1] = cols
        right = left
        value = 0
        paddedImg = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, borderType, None, value)
        paddedImg = cv2.resize(paddedImg, dim, interpolation=cv2.INTER_AREA)
        paddedImg = paddedImg/255
        kanjis[i].append(paddedImg)

    kanjiNPY = np.array(kanjis[i], dtype='float32')
    kanjiNPY = np.reshape(kanjis[i], (len(kanjis[i]), -1))
    train[i::len(img)] = kanjiNPY
    train_labels[i::len(img)] = labels[i]


np.save('train', train)
np.save('train_labels', train_labels)
