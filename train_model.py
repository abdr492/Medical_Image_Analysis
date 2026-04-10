import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from load_preview import prepare_data_generators
import os

def build_cnn_model(input_shape=(256, 256, 1)):
    """
    Builds the CNN model architecture.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5), # Prevent overfitting
        Dense(3, activation='softmax') # Multi-class classification: Normal, Pneumonia, Tuberculosis
    ])
    
    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Define data directory
    # IMPORTANT: Update this path to where your 'chest_xray/train' folder is located
    TRAIN_DATA_DIR = "chest_xray/train"
    
    import argparse

    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    if os.path.exists(TRAIN_DATA_DIR):
        print("Data directory found. Preparing data generators...")
        train_gen, val_gen = prepare_data_generators(TRAIN_DATA_DIR)
        
        print("Building model...")
        model = build_cnn_model()
        model.summary()
        
        print(f"Starting training for {args.epochs} epochs...")
        
        # Callbacks
        checkpoint = ModelCheckpoint("medical_ai_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
        
        try:
            # Train Model
            history = model.fit(
                train_gen,
                epochs=args.epochs,
                validation_data=val_gen,
                callbacks=[checkpoint, early_stop]
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model state...")
        
        # Final Save (Mental backup)
        model.save("medical_ai_model.h5")
        print("Model saved as 'medical_ai_model.h5'")
        
    else:
        print(f"Error: Training data directory '{TRAIN_DATA_DIR}' not found.")
        print("Please download the dataset and extract it to the project folder.")
