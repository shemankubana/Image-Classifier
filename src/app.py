import streamlit as st
import requests
import os
import json
import base64
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This is a temporary solution to import modules from the src directory.
# In a production environment, you would package these.
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.preprocessing import load_processed_data

# --- Streamlit UI Configuration ---
# MUST be the first Streamlit command in the script
st.set_page_config(
    page_title="MLOps CIFAR-10 Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
# CORRECTED: The FastAPI endpoint is /retrain/trigger, not /trigger_retrain
RETRAIN_TRIGGER_ENDPOINT = f"{API_BASE_URL}/retrain/trigger"

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Custom CSS for a Unique Look & Feel ---
st.markdown("""
<style>
    /* Dark theme for a modern feel */
    body {
        color: #fff;
        background-color: #0e1117;
    }
    
    /* Main container styling */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    .css-1d3f3o8, .css-1lcbmhc {
        background-color: #1a1a1a;
        color: #f0f0f0;
    }

    /* Titles and Headers */
    .css-h5gq8t {
        color: #ff4b4b; /* A vibrant red accent */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Buttons with a subtle hover effect */
    .stButton>button {
        color: #f0f0f0;
        background-color: #333;
        border: 1px solid #ff4b4b;
        border-radius: 8px;
        transition: background-color 0.2s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: #0e1117;
        transform: translateY(-2px);
    }
    
    /* Metrics for a clean look */
    [data-testid="stMetric"] {
        background-color: #262626;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        color: #f0f0f0;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricLabel"] > div {
        color: #ff4b4b;
    }

    /* Containers and blocks with rounded corners and spacing */
    .stContainer {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #262626;
        border: 1px solid #444;
    }
    
    /* Warning and Success boxes for a more human feel */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    .stAlert-info {
        border-left-color: #00b0ff !important;
    }
    .stAlert-success {
        border-left-color: #4CAF50 !important;
    }
    .stAlert-warning {
        border-left-color: #ff9800 !important;
    }
    .stAlert-error {
        border-left-color: #f44336 !important;
    }
    
    /* Custom icons for status */
    .status-icon-online {
        color: #4CAF50;
    }
    .status-icon-offline {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions to Interact with FastAPI ---

def get_health_status():
    """Fetches health status from the FastAPI backend."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        return {"status": "error", "message": "Backend unavailable", "model_loaded": False}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"HTTP Error: {e}", "model_loaded": False}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected Error: {e}", "model_loaded": False}

def predict_image_via_api(image_file):
    """Sends an image file to the FastAPI /predict endpoint."""
    try:
        image_file.seek(0)
        files = {'file': (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.error("Could not connect to the API. Is your service running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed with HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None

def trigger_retraining_via_api():
    """Triggers retraining via the FastAPI /retrain/trigger endpoint."""
    st.info("Sending retraining trigger to backend...")
    try:
        response = requests.post(RETRAIN_TRIGGER_ENDPOINT, timeout=300)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.error("Could not connect to the API. Is your service running?")
        return {"status": "error", "message": "Backend unavailable"}
    except requests.exceptions.HTTPError as e:
        st.error(f"Retraining trigger failed with HTTP error {e.response.status_code}: {e.response.text}")
        return {"status": "error", "message": f"HTTP Error: {e.response.text}"}
    except Exception as e:
        st.error(f"An unexpected error occurred while triggering retraining: {e}")
        return {"status": "error", "message": f"Unexpected Error: {e}"}


# --- Streamlit UI Layout ---

st.sidebar.title("🤖 Navigation & Status")
page_selection = st.sidebar.radio("Go to", ["Dashboard", "Predict Image", "Retrain Model"])

st.sidebar.markdown("---")
st.sidebar.header("API Health")

health_data = get_health_status()
# CORRECTED: The FastAPI health endpoint returns "healthy", not "ok"
if health_data.get('status') == 'healthy':
    st.sidebar.markdown(f"**Backend:** <span class='status-icon-online'>🟢</span> Live", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Model:** <span class='status-icon-online'>✅</span> Loaded", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"**Backend:** <span class='status-icon-offline'>🔴</span> Error", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Model:** <span class='status-icon-offline'>❌</span> Not Loaded", unsafe_allow_html=True)
    st.sidebar.warning("API is unreachable. Check your service deployment.")

# --- Main Content Area ---

if page_selection == "Dashboard":
    st.header("✨ The MLOps Dashboard")
    st.markdown("""
    Welcome to the brain of our operation! This dashboard is where we monitor our system's health,
    explore our dataset, and understand our model's performance. It’s the central hub for our MLOps journey.
    """)

    st.subheader("Current System Status")
    col1, col2 = st.columns(2)
    with col1:
        # CORRECTED: The FastAPI health endpoint returns "healthy", not "ok"
        st.metric(label="API Health", value="Online" if health_data.get('status') == 'healthy' else "Error")
    with col2:
        st.metric(label="Model Loaded", value="Yes" if health_data.get('model_loaded') else "No")

    # --- Data Insights Section (Humanized) ---
    st.markdown("---")
    st.subheader("📊 Data Insights: What's the story behind the data?")
    st.write("A good model starts with a good understanding of the data. Here's a look at what makes our dataset tick.")
    
    # CORRECTED: Using load_processed_data to get the data for visualizations
    try:
        _, y_train, _, _, _, _ = load_processed_data()
        # The labels returned are one-hot encoded, so we need to convert them to a flat array
        train_labels_flat = np.argmax(y_train, axis=1)

        st.markdown("#### Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=train_labels_flat, ax=ax)
        ax.set_title('CIFAR-10 Class Distribution')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Count')
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        st.pyplot(fig)
        st.info("""
        **Insight:** The dataset is beautifully balanced! Each of the 10 classes has a roughly equal number of images. This is fantastic because it means our model won't be biased towards one class, leading to a fair and robust classifier.
        """)
        
        st.markdown("---")
        st.markdown("#### Sample Images")
        # Load the data again to get the images, as load_processed_data returns all data
        x_train, _, _, _, _, _ = load_processed_data()
        sample_indices = np.random.choice(len(x_train), 10, replace=False)
        cols = st.columns(10)
        for i, idx in enumerate(sample_indices):
            with cols[i]:
                # Reshape if necessary, though Streamlit usually handles this
                image_to_display = x_train[idx]
                if image_to_display.shape[-1] == 1:
                    image_to_display = image_to_display.squeeze()
                
                st.image(image_to_display, caption=CLASS_NAMES[train_labels_flat[idx]], width=64)
        st.info("""
        **Insight:** Here's a glimpse of the model's challenge: classifying these tiny 32x32 images. They're pixelated and abstract, but our human brains can still see a car or a dog. Our CNN model must learn to do the same, extracting key features to make sense of these low-resolution visuals.
        """)
        
    except Exception as e:
        st.error(f"Could not generate visualizations: {e}. Please ensure the data preprocessing script has been run.")


elif page_selection == "Predict Image":
    st.header("🔮 Your Image, Our Prediction")
    st.markdown("""
    Want to see the magic in action? Upload an image, and our deployed model will tell you what it sees!
    """)
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image for Prediction', use_column_width=True)
        
        if st.button("🚀 Predict!"):
            with st.spinner("Talking to the model... This might take a moment."):
                prediction_result = predict_image_via_api(uploaded_file)

            if prediction_result:
                st.balloons()
                st.success("Prediction Complete!")
                st.metric(label="Predicted Class", value=prediction_result['predicted_class_name'])
                
                st.subheader("Confidence Scores:")
                probabilities_df = pd.DataFrame(prediction_result['probabilities'].items(),
                                                columns=['Class', 'Probability']).sort_values(by='Probability', ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Probability', y='Class', data=probabilities_df, ax=ax)
                ax.set_title('Model Confidence per Class')
                st.pyplot(fig)


elif page_selection == "Retrain Model":
    st.header("🔄 The Retraining Engine")
    st.markdown("""
    This is where we improve our model over time. By feeding it new data, we can adapt it to new patterns and fix its past mistakes.
    """)
    st.markdown("---")
    
    st.warning("""
    **Heads up!** Retraining is a computationally intensive task. While this button triggers the process in the background to keep our API responsive, it will take several minutes to complete.
    """)

    if st.button("🔄 Trigger Model Retraining"):
        with st.spinner("Starting the training process... Grab a coffee! This will run in the background."):
            retrain_response = trigger_retraining_via_api()

        if retrain_response and retrain_response.get('status') == 'triggered':
            st.success(f"Retraining triggered successfully! {retrain_response.get('message')}")
            st.info("Check your service logs on Render for live retraining progress. The model will update automatically once complete.")
        else:
            st.error(f"Failed to trigger retraining: {retrain_response.get('message', 'Unknown error')}")

    st.markdown("---")
    st.subheader("Notes from the MLOps Engineer:")
    st.write("""
    In a real-world MLOps pipeline, we don't just hit a button. New data would be automatically ingested, validated, and used to trigger a fully orchestrated training job. A dedicated 'model registry' would then version and track our new model before it's deployed, ensuring we have full control and traceability.
    """)
