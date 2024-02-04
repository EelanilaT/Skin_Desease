from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np

app = Flask(__name__)

# Ensure the 'uploads' directory exists
uploads_dir = os.path.join(app.root_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Configure uploads folder
app.config['UPLOADS_DEFAULT_DEST'] = uploads_dir
app.config['UPLOADED_IMAGES_DEST'] = uploads_dir
app.config['UPLOADED_IMAGES_ALLOW'] = set(['jpg', 'jpeg', 'png', 'gif'])

# Load models
skin_cancer_model = load_model('D:\SLIIT\RP\skin_cancer_inception_keras\skin_cancer_inception_keras')  # Replace with your actual skin cancer model
other_model = load_model('D:\SLIIT\RP\Model\skin_cancer_inception_keras')  # Replace with your other model

# Function to process the uploaded image for skin cancer model
def process_skin_cancer_image(image_path):
    # Add processing logic specific to the skin cancer model
    pass

# Function to process the uploaded image for the other model
def process_other_model_image(image_path):
    # Add processing logic specific to the other model
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skin_cancer.html')

@app.route('/other_model')
def other_model():
    return render_template('other_model.html')

@app.route('/upload_skin_cancer', methods=['POST'])
def upload_skin_cancer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        filename = secure_filename(file.filename)
        file.save('uploads/' + filename)
        result = process_skin_cancer_image('uploads/' + filename)

        # Map model prediction to class names
        class_names = ['Class1', 'Class2', 'Class3']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})

@app.route('/upload_other_model', methods=['POST'])
def upload_other_model():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        filename = secure_filename(file.filename)
        file.save('uploads/' + filename)
        result = process_other_model_image('uploads/' + filename)

        # Map model prediction to class names
        class_names = ['Class1', 'Class2', 'Class3']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)