from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'model/crop_disease_model_tf.keras'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define your class labels (same order as during training)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__healthy'
]


UPLOAD_FOLDER = 'static/uploads'
UNKNOWN_FOLDER = 'static/unknown'
THRESHOLD = 0.6  # Confidence threshold

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)


# Preprocessing function
def preprocess_image(image_path):
    """Helper function to preprocess the image"""
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/disease-prediction')
def prediction_page():
    return render_template('disease_prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('disease_prediction.html', prediction="No file selected")

    file = request.files['image']
    if file.filename == '':
        return render_template('disease_prediction.html', prediction="No file selected")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    uploaded_image = filename  # for display in template

    try:
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)

        # ✅ Check model output shape matches class_names
        if predictions.ndim != 2 or predictions.shape[1] != len(class_names):
            raise ValueError(
                f"Model output mismatch: expected {len(class_names)} classes but got {predictions.shape[1]}"
            )

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])

        # ✅ Handle low-confidence images
        if confidence < THRESHOLD:
            # Move file to unknown folder
            unknown_path = os.path.join(UNKNOWN_FOLDER, filename)
            shutil.move(filepath, unknown_path)
            # Show closest known class too
            predicted_class = class_names[predicted_index]
            result = f"Closest match: {predicted_class} (confidence: {confidence*100:.2f}%)"
        else:
            predicted_class = class_names[predicted_index]
            result = f"{predicted_class} ({confidence*100:.2f}% confidence)"

        return render_template(
            'disease_prediction.html',
            prediction=result,
            uploaded_image=uploaded_image
        )

    except Exception as e:
        # ✅ Catch any runtime errors safely
        return render_template(
            'disease_prediction.html',
            prediction=f"Error during prediction: {str(e)}",
            uploaded_image=uploaded_image
        )


if __name__ == '__main__':
    app.run(debug=True)
