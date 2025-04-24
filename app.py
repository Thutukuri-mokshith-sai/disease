import numpy as np
import tensorflow as tf
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import nltk
import spacy
from difflib import get_close_matches
import logging
from werkzeug.exceptions import BadRequest
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download NLTK data if not already present
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load disease dataset with error handling
try:
    with open('disease.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'disease.json' file not found.")
    data = {}
except json.JSONDecodeError:
    print("Error: Failed to decode 'disease.json'. Please check the file format.")
    data = {}
# Extract all disease names and create a map of disease names to full keys (e.g., "Apple Apple scab")
disease_keys = list(data.keys())
disease_names_map = {}

for key in disease_keys:
    # Ensure that the key has a space before attempting to split it
    if " " in key:
        disease_names_map[key.split(" ", 1)[1].lower()] = key
    else:
        disease_names_map[key.lower()] = key

# Crop diseases dictionary
diseases_of_crops = {
    "Apple": ["Apple scab", "Black rot", "Cedar apple rust", "healthy"],
    "Banana": ["Cordana", "Panama Disease", "Healthy", "Yellow and Black Sigatoka"],
    "Beans": ["angular leaf spot", "bean rust", "healthy"],
    "Corn (maize)": ["Common rust", "Northern Leaf Blight", "healthy"],
    "Orange": ["Citrus Canker Diseases Leaf", "Citrus Nutrient Deficiency Yellow Leaf", "Healthy Leaf", "Multiple Diseases Leaf", "Young Healthy Leaf"],
    "Potato": ["Early blight", "Late blight", "healthy"],
    "Tomato": ["Bacterial spot", "Early blight", "Late blight", "Leaf Mold", "Septoria leaf spot", "Spider mites Two spotted spider mite", "Target Spot", "Yellow Leaf Curl Virus", "healthy", "mosaic virus"],
    "banana": ["Healthy", "Yellow and Black Sigatoka"],
    "brinjal": ["Healthy Leaf", "Insect Pest Disease", "Leaf Spot Disease", "Mosaic Virus Disease", "Small Leaf Disease", "White Mold Disease", "Wilt Disease"],
    "chilli": ["healthy", "leaf curl", "leaf spot", "whitefly", "yellowish"],
    "cotton": ["bacterial blight", "curl virus", "fussarium wilt", "healthy"],
    "guava": ["Canker", "Dot", "Healthy", "Mummification", "Rust"],
    "mango": ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"],
    "paddy": ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_blast", "leaf_scald", "narrow_brown_spot"],
    "soyabean": ["Mossaic Virus", "Southern blight", "Sudden Death Syndrone", "Yellow Mosaic", "bacterial blight", "brown spot", "crestamento", "ferrugen", "powdery mildew", "septoria"],
    "sugarcane": ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"],
    "watermelon": ["anthracnose", "downy mildew", "healthy", "mosaic virus"]
}

# Global state to track current disease
current_disease = None
current_crop = None

# Utility: best match for disease name using fuzzy matching
def find_best_match(user_input, keys):
    matches = get_close_matches(user_input, keys, n=1, cutoff=0.4)
    return matches[0] if matches else None

# Get diseases by crop
def get_diseases_by_crop(crop_name):
    crop_name = crop_name.lower()
    
    # Check if the crop exists in the diseases_of_crops dictionary
    if crop_name.capitalize() in diseases_of_crops:
        diseases = diseases_of_crops[crop_name.capitalize()]
        response = f"Here are the diseases found in {crop_name.capitalize()}:\n"
        response += "\n".join(diseases)
        response += "\n\nYou can ask for more info about any specific disease."
    else:
        response = f"No diseases found for the crop '{crop_name}'. Please check the name or ask about another crop."
    return response

# Function to find the disease by symptoms using SpaCy
def find_disease(user_input):
    user_input = user_input.lower()

    # Use SpaCy to extract possible disease phrases
    doc = nlp(user_input)
    tokens = ",".join([token.text for token in doc])

    # Try fuzzy match on disease name only (without crop)
    match = find_best_match(tokens, disease_names_map.keys())
    if match:
        return disease_names_map[match]  # return the full key like "Apple Apple scab"

    return None
def get_answer(user_input):
    global current_disease, current_crop
    user_input = user_input.lower()

    # Greeting
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good evening", "howdy"}
    if any(greet in user_input for greet in greetings):
        return "Welcome! How can I assist you with crop diseases?"

    # Check for crop-related queries like "tomato diseases"
    if "disease" in user_input or "diseases" in user_input:
        for crop in diseases_of_crops:
            if crop.lower() in user_input:
                current_crop = crop.lower()
                current_disease = None  # reset disease when crop changes
                return get_diseases_by_crop(crop)

    # Try to identify disease name only if none is selected yet
    if current_disease is None:
        identified_disease = find_disease(user_input)
        if identified_disease:
            current_disease = identified_disease
            return f"You're referring to {current_disease}. What would you like to know about it? (e.g., symptoms, treatments, prevention, organic alternatives, pathogen)"

    # If follow-up question like "symptoms" is asked
    if current_disease:
        disease_info = data.get(current_disease, {})

        if "symptom" in user_input:
            return "\n".join(disease_info.get("symptoms", ["No symptom data available."]))

        elif "treatment" in user_input:
            treatments = disease_info.get("treatments", [])
            if treatments:
                return "\n\n".join([f"Name: {t['name']}\nDosage: {t['dosage']}\nApplication: {t['application']}" for t in treatments])
            return "No treatment data available."

        elif "pathogen" in user_input:
            return f"Pathogen: {disease_info.get('pathogen', 'Unknown')}"

        elif "organic" in user_input:
            organic = disease_info.get("organic_alternatives", [])
            if organic:
                return "\n\n".join([f"Name: {o['name']}\nDosage: {o['dosage']}\nApplication: {o['application']}" for o in organic])
            return "No organic alternatives available."

        elif "prevention" in user_input:
            return "\n".join(disease_info.get("prevention", ["No prevention data available."]))

        elif "safety" in user_input:
            return "\n".join(disease_info.get("safety_tips", ["No safety tips available."]))

        elif any(word in user_input for word in ["details", "info", "information", "about", "describe"]):
            response = f"Information about {current_disease}:\n\n"
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
            return f"What would you like to know about {current_disease}? (e.g., symptoms, treatments, prevention, organic alternatives, pathogen)"

    # If user only gave symptoms or vague text without a known disease
    return "Please provide a disease name or describe the symptoms or crop to get started."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Input validation
        user_input = request.json.get('message', '').strip()
        if not user_input:
            raise BadRequest("No message provided.")
        
        response = get_answer(user_input)
        return jsonify({"response": response})

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500# Preprocess image
def preprocess_image(img_data, target_size=(224, 224)):
    img = Image.open(io.BytesIO(img_data))
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# TFLite prediction
def predict_with_tflite(img_data,
                        tflite_model_path="disease_model.tflite",
                        label_map_path="label_map.json",
                        disease_info_path="disease.json"):

    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
        inv_map = {v: k for k, v in label_map.items()}

    with open(disease_info_path, 'r') as f:
        disease_info = json.load(f)

    img_array = preprocess_image(img_data)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]

    predicted_label = inv_map[predicted_index]
    result = {
        "label": predicted_label,
        "confidence": confidence * 100
    }

    if predicted_label in disease_info:
        info = disease_info[predicted_label]
        result["disease_details"] = info

    return result

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    img_data = image_file.read()

    try:
        result = predict_with_tflite(img_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)  
