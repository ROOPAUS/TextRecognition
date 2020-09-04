from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow
import pandas as pd

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

# for matrix math
import numpy as np

# for regular expressions, saves time dealing with string data
import re

# system level operations (like loading files)
import sys

# for reading operating system data
import os

# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request

# initalize our flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './models/'

json_file = open(MODEL_PATH+'model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_PATH +"model.h5")

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'uploads')


def model_predict_mnit(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28))
    img = img.convert('L')
    sample1 = np.fliplr(img)
    img = np.rot90(sample1)
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    x = x.reshape(1,28,28,1)
    x = x/255
    
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

def model_predict(x, model):
    # Preprocessing the image
    xy = image.img_to_array(x)
    # xy = np.true_divide(xy, 255)
    xy = np.expand_dims(xy, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    xy = preprocess_input(xy, mode='caffe')

    preds = model.predict(xy)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        filename = image.filename
        file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        image.save(file_path)
        preds = model_predict_mnit(file_path, model)
        # Process your result for human
        pred_class = preds.argmax(axis=-1)
        dataframe = pd.read_csv("C:/Users/roopa/Downloads/Labels.csv", header=None)
        dictionary = {}
        dataframe = np.asarray(dataframe)
        for i in range(len(dataframe[:,0])):
            dictionary[dataframe[i,0]] = dataframe[i,1]
            print(dictionary.get(pred_class[0]))
        return dictionary.get(pred_class[0])
    return None


if __name__ == '__main__':
    model=loaded_model
    app.run(host='0.0.0.0', port=5000, debug=True)