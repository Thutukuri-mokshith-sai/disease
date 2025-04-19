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
img_height, img_width = 224, 224  # Match the preprocess size used earlier
model_path = "plant_disease_model.tflite"
label_map_path = "label_map.json"
disease_info_path = "disease.json"

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Load model
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    app.logger.info("✅ Model loaded.")
except Exception as e:
    app.logger.error(f"❌ Error loading model: {e}")

# Load label map
try:
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    inv_map = {v: k for k, v in label_map.items()}
    app.logger.info("✅ Label map loaded.")
except Exception as e:
    app.logger.error(f"❌ Error loading label map: {e}")

# Load disease info
try:
    with open(disease_info_path, "r") as f:
        disease_data = json.load(f)
    app.logger.info("✅ Disease information loaded.")
except Exception as e:
    app.logger.error(f"❌ Error loading disease info: {e}")

# Util function: Preprocess image
def preprocess_image(file_stream):
    image = Image.open(file_stream).convert('RGB')
    image = image.resize((img_width, img_height))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']

    try:
        img_array = preprocess_image(io.BytesIO(image_file.read()))
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        predicted_label = label_map.get(predicted_index, "Unknown")

        app.logger.debug(f"🔍 Predicted: {predicted_label} ({confidence * 100:.2f}%)")

        disease_info = disease_data.get(predicted_label, {})

        return jsonify({
            "class": predicted_label,
            "confidence": f"{confidence * 100:.2f}%",
            "info": disease_info
        })

    except Exception as e:
        app.logger.error(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Error during prediction"}), 500

# Run server
if __name__ == '__main__':
    app.run(debug=True)
