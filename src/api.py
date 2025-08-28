import sys
import os

# This block is crucial for local development to ensure imports work correctly.
# It adds the parent directory (project root) to the system path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
from datetime import datetime
import numpy as np
import asyncio
import uuid
import tensorflow as tf

# Define the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Import from the src module directly
from src.prediction import predict_image, load_inference_model, CLASS_NAMES
from src.preprocessing import load_and_preprocess_data
from src.model import build_model, train_model

app = FastAPI(
    title="MLOps CIFAR-10 Image Classifier API",
    description="API for predicting CIFAR-10 images and triggering model retraining.",
    version="1.0.0"
)

# --- API Models for Request/Response ---

class PredictionResponse(BaseModel):
    predicted_class_index: int
    predicted_class_name: str
    probabilities: dict[str, float]

class RetrainTriggerResponse(BaseModel):
    message: str
    status: str
    retraining_initiated_at: str
    model_update_behavior: str = "new_model_loaded_in_memory"

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    model_loaded: bool

# --- Global state for model and retraining ---
is_retraining_in_progress = False
INCOMING_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'incoming_for_retraining')
os.makedirs(INCOMING_DATA_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """
    Load the model when the FastAPI application starts.
    """
    try:
        load_inference_model()
        print("Model pre-loaded during API startup.")
    except Exception as e:
        print(f"Failed to pre-load model during startup: {e}. API may not function correctly.")


async def perform_retraining_task():
    """
    Performs the full model retraining process.
    """
    global is_retraining_in_progress
    print("\n--- Background retraining task started ---")
    try:
        print("Retraining: Loading existing preprocessed data (or generating if not exists)...")
        # Using the corrected function from src.preprocessing
        x_train, y_train, x_val, y_val, _, _ = load_processed_data()
        
        print("Retraining: Data loaded and preprocessed.")

        print("Retraining: Building and compiling a new model instance...")
        # Using the corrected function from src.model
        new_model_instance = build_model()
        
        print("Retraining: Starting model training...")
        tf.keras.backend.clear_session()
        # Using the corrected function from src.model
        history, saved_model_path = train_model(new_model_instance, x_train, y_train, x_val, y_val)
        print("Retraining: Model training completed.")
        
        # Save the new model to disk (simulating model versioning/updates)
        # Note: Your train_model function already saves the model,
        # but this reloads it just in case.
        new_model_instance.save(os.path.join(PROJECT_ROOT, 'models', 'cifar10_cnn_model.h5'))
        print(f"Retraining: Newly trained model saved to {os.path.join(PROJECT_ROOT, 'models', 'cifar10_cnn_model.h5')}.")

        # Reload the newly trained model for inference
        load_inference_model()
        print("Retraining: Newly trained model loaded into API for live inference.")
        print("--- Background retraining task completed successfully ---")

    except Exception as e:
        print(f"--- Background retraining task failed: {e} ---")
    finally:
        is_retraining_in_progress = False
        print("Retraining status flag reset to False.")


@app.get("/health", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    """
    Checks the health status of the API and if the ML model is loaded.
    """
    model = load_inference_model()
    model_status = True if model is not None else False
    return {
        "status": "healthy",
        "message": "API is operational.",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_status
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Image Class")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and returns the predicted class
    and class probabilities.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    model = load_inference_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot make prediction.")

    try:
        image_bytes = await file.read()
        image_stream = io.BytesIO(image_bytes)
        prediction_result = predict_image(image_stream)
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/retrain/trigger", response_model=RetrainTriggerResponse, summary="Trigger Model Retraining")
async def trigger_retraining_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers the model retraining process in the background.
    """
    global is_retraining_in_progress

    if is_retraining_in_progress:
        raise HTTPException(status_code=409, detail="Retraining already in progress. Please wait.")

    is_retraining_in_progress = True
    background_tasks.add_task(perform_retraining_task)

    return {
        "message": "Model retraining successfully triggered in background. Check API logs for progress.",
        "status": "triggered",
        "retraining_initiated_at": datetime.now().isoformat(),
        "model_update_behavior": "new_model_loaded_in_memory"
    }
