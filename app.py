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

# # Load models
# skin_cancer_model = load_model('D:\\RP\\Nila\\skin_cancer_inception')
# Benign_Tumors_model = load_model('D:\RP\model\Ayingaran\ResNet50')
# skin_rashes_Model = load_model('D:\RP\Sharujan\a\skin_condition_model')  

skin_cancer_model = load_model('D:\\RP\\Nila\\skin_cancer_inception')
Benign_Tumors_model = load_model('D:\\RP\\model\\Ayingaran\\ResNet50')
skin_rashes_Model = load_model('D:\\RP\\Sharujan')
Inflam_Skin_Model= load_model('D:\RP\Dalax')
# Function to process the uploaded image for skin cancer model
def process_skin_cancer_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = skin_cancer_model.predict(img_array)
    return result

# Function to process the uploaded image for the Benign_Tumors model
def process_Benign_Tumors_model_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = Benign_Tumors_model.predict(img_array)
    return result

# Function to process the uploaded image for the Third_Model
def process_skin_rashes_Model_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = skin_rashes_Model.predict(img_array)
    return result

# Function to process the uploaded image for the Inflam
def process_inflam_skin_Model_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = Inflam_Skin_Model.predict(img_array)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skin_cancer.html')

@app.route('/other_model')
def Benign_Tumors():
    return render_template('other_model.html')

@app.route('/third_model')
def third_model():
    return render_template('third_model.html')

@app.route('/inflam_model')
def inflam_model():
    return render_template('inflam_model.html')

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
        class_names = ['Basal_Cell_Carcinoma', 'Melanoma', 'NormalSkin']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})

@app.route('/upload_benign_tumors', methods=['POST'])
def upload_benign_tumors():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        filename = secure_filename(file.filename)
        file.save('uploads/' + filename)
        result = process_Benign_Tumors_model_image('uploads/' + filename)

        # Map model prediction to class names
        class_names = ['Melanocytic_Nevi', 'Benign_Keratosis_like_Lesions', 'NormalSkin']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})

@app.route('/upload_skin_rashes_Model', methods=['POST'])
def upload_third_model():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        filename = secure_filename(file.filename)
        file.save('uploads/' + filename)
        result = process_skin_rashes_Model_image('uploads/' + filename)

        # Map model prediction to class names
        class_names = ['WartsMolluscum', 'UrticariaHives','NormalSkin']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})
    
@app.route('/upload_inflam_model', methods=['POST'])
def upload_inflam_model():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image file selected'})

    if file:
        filename = secure_filename(file.filename)
        file.save('uploads/' + filename)
        result = process_inflam_skin_Model_image('uploads/' + filename)

        # Map model prediction to class names
        class_names = ['Akne', 'Eczema','NormalSkin']
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'result': predicted_class})    

if __name__ == '__main__':
    app.run(debug=True)
