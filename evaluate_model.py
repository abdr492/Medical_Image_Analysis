from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def evaluate_model(model_path, test_dir):
    """
    Evaluates the trained model on the test dataset.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    # Load trained model
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")

    # Prepare test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        classes=['normal', 'pneumonia', 'tuberculosis'] # Explicitly select classes
    )

    # Predict on test images
    print("Predicting on test images...")
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1) # Get index of max probability
    y_true = test_generator.classes

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred_classes)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia', 'Tuberculosis'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    
    # Save the plot instead of showing it (which blocks execution)
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    MODEL_PATH = "medical_ai_model.h5"
    TEST_DATA_DIR = "chest_xray/test" # Ensure this path is correct
    
    evaluate_model(MODEL_PATH, TEST_DATA_DIR)
