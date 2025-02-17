import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Model and label setup
script_dir = os.path.dirname(os.path.realpath(__file__))
covid_model_path = os.path.join(script_dir, 'model_final.h5')
wound_labels = ["Abrasions", "Bruises", "Cut", "Laceration", "Stab wound"]

# Load the trained model
model = load_model(covid_model_path)

@app.route('/')
def home():
    return "Wound Classifier API is running"

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']

    try:
        # Load and preprocess the image
        img = image.load_img(img_file, target_size=(224, 224))  # Adjust size if needed
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = wound_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5000)
