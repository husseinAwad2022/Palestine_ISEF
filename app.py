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

@app.route('/classify', methods=['GET'])
def classify_image():
    image_path = request.args.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid or missing image path"}), 400

    # Load and preprocess image for the wound classifier
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict using Keras model
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    result = wound_labels[predicted_class]
    
    return jsonify({"image": image_path, "prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5000)
