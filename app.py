import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
import os
import io
from PIL import Image
import json
import nltk
import spacy
from difflib import get_close_matches
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

# Download necessary NLTK data
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load disease dataset
with open('disease.json', 'r') as f:
    data = json.load(f)

# Extract all disease names
disease_keys = list(data.keys())

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Greetings
greetings = {"hi", "hello", "hey", "greetings", "good morning", "good evening", "howdy"}

# Global state
current_disease = None

# Utility function to find best match for disease name
def find_best_match(user_input, keys):
    matches = get_close_matches(user_input, keys, n=1, cutoff=0.4)
    return matches[0] if matches else None

# Function to find disease by symptom
def find_disease_by_symptom(user_input):
    user_tokens = set(nltk.word_tokenize(user_input.lower()))
    for disease, details in data.items():
        for symptom in details.get("symptoms", []):
            symptom_tokens = set(nltk.word_tokenize(symptom.lower()))
            if user_tokens & symptom_tokens:  # if there's any overlap
                return disease
    return None
def get_answer(user_input):
    global current_disease
    user_input = user_input.lower()

    # Greeting
    if any(greet in user_input for greet in greetings):
        return "Welcome! How can I assist you with crop diseases?"

    # Check if user asked for diseases of a specific crop
    if "disease" in user_input or "diseases" in user_input:
        for disease_name in disease_keys:
            if any(crop in user_input for crop in disease_name.lower().split()):
                matching_diseases = [d for d in disease_keys if crop in d.lower()]
                if matching_diseases:
                    response = f"The following diseases are found in crops related to '{crop.capitalize()}':\n"
                    response += "\n".join(matching_diseases)
                    response += "\n\nYou can ask about any of these to know more."
                    return response

    # If a disease name is directly mentioned
    identified_disease = find_best_match(user_input, disease_keys)

    # Try to infer disease if none is directly matched
    if not identified_disease:
        identified_disease = find_disease_by_symptom(user_input)

    # Update current_disease if a new disease is identified
    if identified_disease:
        current_disease = identified_disease

    # If no current_disease is known, ask for more input
    if not current_disease:
        return "Please provide the disease name or describe the symptoms."

    disease_info = data.get(current_disease, {})

    # Respond based on user query
    if "symptom" in user_input:
        return "\n".join(disease_info.get("symptoms", ["No symptom data available."]))

    elif "treatment" in user_input:
        treatments = disease_info.get("treatments", [])
        if treatments:
            return "\n\n".join([f"Name: {t['name']}\nDosage: {t['dosage']}\nApplication: {t['application']}" for t in treatments])
        else:
            return "No treatment data available."

    elif "pathogen" in user_input:
        return f"Pathogen: {disease_info.get('pathogen', 'Unknown')}"

    elif "organic" in user_input:
        organic = disease_info.get("organic_alternatives", [])
        if organic:
            return "\n\n".join([f"Name: {o['name']}\nDosage: {o['dosage']}\nApplication: {o['application']}" for o in organic])
        else:
            return "No organic alternatives available."

    elif "prevention" in user_input:
        return "\n".join(disease_info.get("prevention", ["No prevention data available."]))

    elif "safety" in user_input:
        return "\n".join(disease_info.get("safety_tips", ["No safety tips available."]))

    elif any(keyword in user_input for keyword in ["details", "info", "information", "about", "describe"]):
        response = f"Here is the information about {current_disease}.\n"
        response += "Symptoms:\n" + "\n".join(disease_info.get("symptoms", [])) + "\n\n"
        response += "Prevention:\n" + "\n".join(disease_info.get("prevention", [])) + "\n\n"
        response += "Treatments:\n"
        treatments = disease_info.get("treatments", [])
        if treatments:
            response += "\n\n".join([f"Name: {t['name']}\nDosage: {t['dosage']}\nApplication: {t['application']}" for t in treatments])
        else:
            response += "No treatment info available."
        return response

    else:
        return f"You're referring to {current_disease}. What would you like to know? (e.g., symptoms, treatments, prevention, organic alternatives, pathogen)"

# API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    print("chat")
    user_input = request.json.get('message', '')
    response = get_answer(user_input)
    return jsonify({"response": response})


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode
