import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

# ------------------------------
# MODEL LOADING
# ------------------------------
MODEL_PATH = "model/crop_disease_model.tf.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# CLASS LABELS
# ------------------------------
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__healthy'
]

THRESHOLD = 0.6  # Confidence Threshold
UNKNOWN_FOLDER = "static/unknown"
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

# ------------------------------
# PREPROCESS FUNCTION
# ------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    return img_array

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict(image):
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)

        # Model output validation
        if predictions.ndim != 2 or predictions.shape[1] != len(class_names):
            return f"Model output mismatch: expected {len(class_names)} classes."

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        predicted_class = class_names[predicted_index]

        # Handle unknown / low-confidence images
        if confidence < THRESHOLD:
            return f"❓ Unknown Image\nClosest Match: {predicted_class} ({confidence*100:.2f}%)"
        else:
            return f"🌿 Disease: {predicted_class}\nConfidence: {confidence*100:.2f}%"

    except Exception as e:
        return f"Error during prediction: {str(e)}"

# ------------------------------
# GRADIO UI
# ------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Crop Disease Detection",
    description="Upload a crop leaf image to detect its disease using a TensorFlow CNN model.",
)

if __name__ == "__main__":
    interface.launch()
