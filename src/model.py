import os
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import load_processed_data
import pickle # New import for saving history

# Define paths relative to the project root
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure the models directory exists

def build_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a Convolutional Neural Network (CNN) model for CIFAR-10 classification.
    Includes Batch Normalization and Dropout for optimization/regularization.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Regularization technique
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=64, model_name='cifar10_cnn_model.h5', history_name='training_history.pkl'):
    """
    Trains the provided Keras model with given training and validation data.
    Includes EarlyStopping and ModelCheckpoint for optimization.
    Saves the best model and its training history.
    """
    print("Compiling model...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compiled.")

    # Callbacks for optimization and saving the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_filepath = os.path.join(MODELS_DIR, model_name)
    model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', save_best_only=True, verbose=1)

    print("Training model...")
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    print("Model training complete.")
    print(f"Best model saved to: {model_filepath}")

    # Save training history
    history_filepath = os.path.join(MODELS_DIR, history_name)
    with open(history_filepath, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_filepath}")

    return history, model_filepath # Return history object as well

if __name__ == "__main__":
    # Load processed data (now includes x_val, y_val)
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data()

    # Define input shape and number of classes
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]

    # Build the model
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    # Train the model using the proper train/validation split
    history, saved_model_path = train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=64)

    # Now, after training, we can load the best model and evaluate it on the truly unseen test set
    final_model = tf.keras.models.load_model(saved_model_path)
    print(f"\nLoaded best model from {saved_model_path}")

    print("\nEvaluating the best model on the TEST set (truly unseen data)...")
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")