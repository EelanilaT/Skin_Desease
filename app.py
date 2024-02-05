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

# Load pre-trained model
model = load_model('D:\RP\model\Ayingaran\ResNet50')  

# Function to process the uploaded image
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust the size based on your model requirements
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        # Ensure the 'uploads' directory exists
        os.makedirs(uploads_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file.save(os.path.join(uploads_dir, filename))
        result = process_image(os.path.join(uploads_dir, filename))

        # Map model prediction to class names
        class_names = ['Melanocytic_nevus', 'Benign_keratosis', 'Normal_skin']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
