import json 
import base64
import io
import pickle
import math
from flask import Flask, request
from imageio import imread
import cv2
import numpy as np

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def procesar():
    b64Im = request.get_json()
    ionicImg = base64toUint8(b64Im['image'])
    imPredict = resize(ionicImg)
    return predict(imPredict)
   
def base64toUint8(b64Im):
    data = base64.b64decode(b64Im)
    # reconstruct image as an numpy array
    return imread(io.BytesIO(data))

def resize(ionicImg):
    borderType = cv2.BORDER_CONSTANT
    #img = cv2.imread(ionicImg, cv2.IMREAD_UNCHANGED)
    #make mask of where the transparent bits are
    trans_mask = ionicImg[:, :, 3] == 0
    #replace areas of transparency with white and not transparent
    ionicImg[trans_mask] = [255, 255, 255, 255]
    gray = cv2.bitwise_not(cv2.cvtColor(ionicImg, cv2.COLOR_BGRA2GRAY))
   
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
    imgRe = cv2.resize(paddedImg, (150, 150), interpolation=cv2.INTER_AREA)
    imgRe = imgRe/255
    
    array = []
    array.append(imgRe)
    array = np.reshape(array, (len(array), -1))
    return array

def predict(imPredict):
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    answer = loaded_model.predict_proba(imPredict)
    answer = np.around(answer, decimals=3)
    
    answer = answer.tolist()
    return json.dumps({'Results': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)