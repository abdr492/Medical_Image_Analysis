# AI-Powered Medical Image Analysis - Pneumonia & Tuberculosis Detection

This project implements an AI system using a Convolutional Neural Network (CNN) to detect **Pneumonia** and **Tuberculosis** from Chest X-Ray images. It features a professional web interface for real-time analysis.

## 🚀 Project Overview
- **Data Loading**: Automatically loads and previews grayscale X-ray images.
- **Preprocessing**: Resizes images to 256x256 and applies data augmentation for robustness.
- **Model Training**: Trained on a dataset of ~16,000 images over **15 Epochs**.
- **Model Architecture**: Multi-class CNN (Normal, Pneumonia, Tuberculosis).
- **Deployment**: Local Flask web application with a modern, responsive UI.

## 🛠️ Project Structure
```
Medical_Image_Analysis/
├── app.py               # Flask Web Application (Backend)
├── train_model.py       # Improved Training Script with Early Stopping
├── evaluate_model.py    # Evaluation Script for Metrics & Confusion Matrix
├── load_preview.py      # Data Preprocessing & Preview Script
├── medical_ai_model.h5  # Trained AI Model (Final Version)
├── requirements.txt     # Python Dependencies
├── run_app.bat          # One-click Application Launcher
├── templates/
│   └── index.html       # Premium Web Interface (Frontend)
└── chest_xray/          # Dataset (Normal, Pneumonia, Tuberculosis)
```

## 🏥 Use Instructions
### 1. Quick Start
Simply double-click the **`run_app.bat`** file.
- It will automatically check for the trained model.
- It will launch the web interface at `http://127.0.0.1:5000`.

### 2. Manual Execution
If you prefer the terminal:
```powershell
# Activate Virtual Environment
.\venv\Scripts\activate

# Run the App
python app.py
```

## 📊 Model Performance
The model has been trained for 15 epochs on a diverse dataset to ensure high confidence and accuracy across all three diagnostic categories:
- **Normal**: Healthy lungs.
- **Pneumonia**: Infection-based fluid in lungs.
- **Tuberculosis**: Specific bacterial infection markers.

## 🎨 Features
- **Modern UI**: Clean, responsive design with "Outfit" typography.
- **Drag & Drop**: Easily upload X-rays directly into the browser.
- **Real-time Prediction**: Get results and confidence scores in seconds.
- **Class Indicators**: Visual color-coding for different diagnostic results.

---
*Disclaimer: This is an AI research project and should not be used as a primary medical diagnostic tool.*
