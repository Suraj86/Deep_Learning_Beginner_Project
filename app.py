from __future__ import division, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Importing Keras Libraries
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# Compulsory for this kind of model
model._make_predict_function()          



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    # Carefully train the model to make the model to predict correctly
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Getting the file from post request
        f = request.files['file']

        # Saving the new file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predicting the model
        preds = model_predict(file_path, model)

        # Processing the Result
        #ImageNet Decode
        pred_class = decode_predictions(preds, top=1)   
        
        # Converting to String
        result = str(pred_class[0][0][1])               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
