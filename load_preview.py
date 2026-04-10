import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_and_display_image(image_path):
    """
    Loads an image from the specified path and displays it in grayscale.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read image.")
        return

    # Show image
    plt.imshow(image, cmap='gray')
    plt.title("Sample X-Ray Image")
    plt.axis('off')
    plt.show()

def prepare_data_generators(data_dir, target_size=(256, 256), batch_size=32):
    """
    Prepares data generators for training and validation with augmentation.
    """
    # Define Image Data Generator for Augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% of data for validation
    )

    # Load training dataset
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        classes=['normal', 'pneumonia', 'tuberculosis'], # Explicitly select classes
        subset='training'
    )

    # Load validation dataset
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        classes=['normal', 'pneumonia', 'tuberculosis'], # Explicitly select classes
        subset='validation'
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    # Define data directory
    data_dir = "chest_xray/train"
    
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Use a real image path for preview
    sample_image_path = os.path.join(data_dir, "pneumonia", "pneumonia-1000.jpg")
    
    if os.path.exists(sample_image_path):
        print(f"Loading sample image from: {sample_image_path}")
        image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
             # Save preview image as requested
             cv2.imwrite("preview_sample.png", image)
             print("SUCCESS: Preview image saved as 'preview_sample.png'")
        else:
             print("ERROR: Could not read sample image with cv2.")
    else:
        # Fallback to search recursively for first pneumonia image if exact file is missing
        # (Since filenames might differ in user's dataset version)
        import glob
        pneumonia_images = glob.glob(os.path.join(data_dir, "pneumonia", "*.jpeg"))
        if pneumonia_images:
             print(f"Loading sample image from: {pneumonia_images[0]}")
             image = cv2.imread(pneumonia_images[0], cv2.IMREAD_GRAYSCALE)
             if image is not None:
                 cv2.imwrite("preview_sample.png", image)
                 print("SUCCESS: Preview image saved as 'preview_sample.png'")
        else:
             print(f"WARNING: No sample pneumonia image found in {os.path.join(data_dir, 'pneumonia')}")

    if os.path.exists(data_dir):
        print("Initializing generators...")
        try:
            train_gen, val_gen = prepare_data_generators(data_dir)
            print("SUCCESS: Data generators created successfully.")
            print(f"Training samples: {train_gen.samples}")
            print(f"Classes: {list(train_gen.class_indices.keys())}")
        except Exception as e:
            print(f"ERROR: Failed to create data generators: {e}")
    else:
        print(f"ERROR: Data directory '{data_dir}' not found.")
