# Use a recent and secure Python base image.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, apply security patches, and clean up.
# We add curl and wget for downloading files, though curl/wget won't be used for your LFS files anymore.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    wget \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install the Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and models.
# These will be populated by the COPY . . command below.
RUN mkdir -p /app/data/test \
    && mkdir -p /app/models

# --- REMOVED THE CURL COMMANDS HERE ---
# The large files (x_test.npy and cifar10_cnn_model.h5)
# should already be in your GitHub repository and will be
# copied into the Docker image by the 'COPY . .' command below.

# Copy the rest of the application code into the container.
# This includes all files from your Git repository's root,
# which now correctly includes your LFS-tracked files.
COPY . .

# Expose the port that your FastAPI application will listen on.
EXPOSE 8000

# Define the command to run your application using Uvicorn.
# IMPORTANT: Use $PORT here, not a hardcoded 8000.
# Also, remove --reload for production deployment.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "10000"]