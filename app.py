from __future__ import division, print_function
from flask import Flask, request, render_template, redirect, url_for ,send_file, send_from_directory
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image as pil_image
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import pickle 
import ast

app = Flask(__name__)

@app.route('/dermitology.html')
def dermitology():
    return render_template('dermitology.html')

@app.route('/skin-predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, Model)
        
        # Get the predicted class index
        pred_class = preds.argmax(axis=-1)[0] if preds.size > 0 else None
        
        # Debug: Check if pred_class is valid
        print("Predicted class index:", pred_class)
        
        # Ensure pred_class is valid
        if pred_class is not None and pred_class in lesion_classes_dict:
            # Get the predicted label from lesion_classes_dict
            predicted_label = lesion_classes_dict[pred_class]
        else:
            predicted_label = "Undefined"

        # Retrieve related images randomly from the predicted class
        image_paths = get_images_by_prediction(pred_class, n=6) if pred_class is not None else []
        
        # Prepare response data
        response_data = {
            'result': predicted_label,
            'images': image_paths
        }
        
        return response_data  # Return JSON response
    return None

# Route to serve images from the images folder
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(images_folder, filename)


if __name__ == '__main__':
    app.run(debug=True)