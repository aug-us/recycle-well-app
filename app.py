from __future__ import division, print_function
# coding=utf-8
import os
from pathlib import Path

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)


path = Path("./path/models")
learn = load_learner(path, 'export_with_aug.pkl')


def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class
    

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files.get('myfile', '')
    #f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path)
    preds = str(preds)
    return render_template('output.html', preds=preds)


if __name__ == '__main__':
    
    app.run()