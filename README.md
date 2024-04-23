# Cat and Dog Image Classifier

This project is designed to classify images as either a cat or a dog using a trained machine learning model. It includes a FastAPI application for handling REST API requests and a Gradio interface for easy interaction via a web interface.

## Project Structure

- `app/` - Contains the source code for the FastAPI and Gradio applications.
  - `utils/` - Contains code with helpers functions.
  - `gradio_main.py` - Contains the Gradio interface code.
  - `fastapi_main.py` - Contains the FastAPI app code.
  - `training.py` - Contains the training code.
- `output/models/` - Contains train models
- `Dockerfile` - Contains the Docker configuration for building container.

## Setup

### Prerequisites

Ensure you have Docker installed on your machine. Docker will be used to build and run the containers for the FastAPI and Gradio applications.

### Building and Running the Application

1. **Build the Gradio Docker Container:**

   Navigate to the root directory of the project and run:

   ```bash
   docker build -t gradio .
   ```

2. **Run the Gradio Docker Container:**

   Navigate to the root directory of the project and run:

   ```bash
   docker run -p 8000:8000 gradio
   ```

### **To test FastAPI endpoint run in command line**

```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@path/to/your/image.jpg" -H "Content-Type: multipart/form-data"
```
