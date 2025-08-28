    #!/bin/bash

    # Exit immediately if a command exits with a non-zero status.
    set -e

    # Starting the FastAPI server with Uvicorn, binding to the port provided by Render
    echo "Starting the FastAPI server with Uvicorn..."
    uvicorn src.api:app --host 0.0.0.0 --port $PORT
    