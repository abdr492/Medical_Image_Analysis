from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "medical_ai_model.h5")

print(f"Checking for model at: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        # Load model without optimizer to save memory for inference
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[SUCCESS] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        model = None
else:
    print(f"[WARNING] Model file NOT FOUND at {MODEL_PATH}")
    model = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure medical_ai_model.h5 exists and is valid."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image file
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Empty file uploaded"}), 400
            
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({"error": "Invalid image format. Please upload a valid X-ray image (JPG, PNG, JPEG)."}), 400
            
        # Preprocess image
        image = cv2.resize(image, (256, 256)) / 255.0
        image = image.reshape(1, 256, 256, 1).astype(np.float32) # Ensure correct type and shape
        
        # Predict
        prediction = model.predict(image)[0]
        class_idx = np.argmax(prediction)
        
        # Result
        classes = ['Normal', 'Pneumonia', 'Tuberculosis']
        result = classes[class_idx]
        confidence = float(prediction[class_idx])
        
        return jsonify({
            "Prediction": result,
            "Confidence": confidence
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500

import threading
import webbrowser

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Running with debug=False to prevent double model loading/reloading issues
    threading.Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
