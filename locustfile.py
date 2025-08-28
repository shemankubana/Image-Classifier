from locust import HttpUser, task, between
import base64
import io
from PIL import Image
import numpy as np

RENDER_API_HOST = "https://cifar10-mlops-api.onrender.com"

PREDICT_ENDPOINT = "/predict"

TEST_IMAGE_BASE64 = ""

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def generate_test_image_base64():
    """
    Generates a dummy 32x32 pixel image and returns it as a base64 encoded string.
    This is used to simulate a prediction request without needing a physical file.
    """
    image = Image.new('RGB', (32, 32), color='red')
    
    byte_io = io.BytesIO()
    image.save(byte_io, format='JPEG')
    image_bytes = byte_io.getvalue()
    
    base64_encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_encoded_string

class APIUser(HttpUser):
    
    host = RENDER_API_HOST
    
    wait_time = between(1, 2)

    def on_start(self):
        """
        Called when a Locust user starts running.
        We use this to prepare the base64 encoded image just once per user.
        """
        global TEST_IMAGE_BASE64
        if not TEST_IMAGE_BASE64:
            TEST_IMAGE_BASE64 = generate_test_image_base64()
            print("Generated test image for flood request simulation.")

    @task
    def predict_image(self):
        """
        Simulates a user sending a prediction request to the API.
        The payload matches what the FastAPI endpoint expects.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "image_base64": TEST_IMAGE_BASE64
        }
        
        self.client.post(PREDICT_ENDPOINT, json=payload, name="/predict")

    @task
    def check_health(self):
        """
        Simulates a user checking the health of the API.
        This is a lightweight GET request.
        """
        self.client.get("/health", name="/health")