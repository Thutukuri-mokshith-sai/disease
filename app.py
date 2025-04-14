import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
import os
import io
from PIL import Image
# Configurations
img_height, img_width = 128, 128
model_path = "plant_disease_model.tflite"  # Ensure this path is correct
label_map_path = "label_map.json"  # Ensure this path is correct
disease_info_path = "disease.json"  # Ensure this path is correct

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Load the TensorFlow Lite model
try:
    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    app.logger.info(f"Model loaded from {model_path}")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")

# Load label map for decoding model predictions
try:
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}  # Convert label_map keys to integers
    app.logger.info(f"Label map loaded from {label_map_path}")
except Exception as e:
    app.logger.error(f"Error loading label map: {e}")

# Load disease information (optional for returning more details about the disease)
try:
    with open(disease_info_path, "r") as f:
        disease_data = json.load(f)
    app.logger.info(f"Disease information loaded from {disease_info_path}")
except Exception as e:
    app.logger.error(f"Error loading disease information: {e}")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug('Received request')

    if 'image' not in request.files:
        app.logger.error('No image provided')
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    
    app.logger.debug('Image received, processing...')

    try:
        # Load image and prepare it for prediction
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((img_width, img_height))  # Resize image
        img_array = np.array(image) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and convert to float32

        # Set the input tensor
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run the inference
        interpreter.invoke()

        # Get prediction results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output_data))

        predicted_label = label_map.get(predicted_index, "Unknown")
        
        app.logger.debug(f'Prediction: {predicted_label}')
        
        # Get additional disease information
        disease_info = disease_data.get(predicted_label, "No data found")

        # Return the result in JSON format
        return jsonify({
            "class": predicted_label,
            "info": disease_info
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Error during prediction"}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode
