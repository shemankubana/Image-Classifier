import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io

# The working directory in the Docker container is /app.
# The 'models' folder is a subdirectory of /app.
# This path is robust inside and outside the container.
# This will be the FINAL correct path.
MODEL_PATH = "models/cifar10_cnn_model.h5"

# Global variable to store the loaded model
_model = None
# Global variable for class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_inference_model():
    """
    Loads the trained Keras model from the specified path.
    Uses a global variable to ensure the model is loaded only once.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
        try:
            _model = load_model(MODEL_PATH)
            print(f"Model '{MODEL_PATH}' loaded successfully for inference.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            _model = None
    return _model

def preprocess_image_for_prediction(image_bytes_io):
    """
    Preprocesses a raw image for model prediction.
    """
    try:
        image = Image.open(image_bytes_io).convert('RGB')
        image = image.resize((32, 32))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_image(image_bytes_io):
    """
    Makes a prediction on a single image provided as a BytesIO object.
    """
    model = load_inference_model()
    if model is None:
        raise RuntimeError("Machine Learning model is not loaded. Cannot make prediction.")

    processed_image = preprocess_image_for_prediction(image_bytes_io)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    probabilities = predictions[0].tolist()

    return {
        "predicted_class_index": int(predicted_class_index),
        "predicted_class_name": predicted_class_name,
        "probabilities": dict(zip(CLASS_NAMES, probabilities))
    }

if __name__ == "__main__":
    print("Running prediction.py directly for testing...")

    # ... (rest of the local test code)

    # For testing, let's create a very simple in-memory image (e.g., solid red)
    # This completely avoids file path issues for the internal test.
    # In the actual FastAPI app, you'll receive real image bytes from an upload.
    test_image_pil = Image.new('RGB', (32, 32), color='red')
    byte_io = io.BytesIO()
    test_image_pil.save(byte_io, format='PNG') # Save to BytesIO object
    byte_io.seek(0) # Rewind the buffer to the beginning

    try:
        prediction_result = predict_image(byte_io) # Pass BytesIO directly
        print("\nPrediction Result:")
        print(prediction_result)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your model 'cifar10_cnn_model.h5' exists in the 'models/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        print("This might be due to an issue with image processing or model inference.")
        # Re-raise the exception to see full traceback if still an issue
        # raise
    print("\nLocal prediction test complete.")