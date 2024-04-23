# Use an official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container

WORKDIR /

RUN apt-get update
RUN apt-get install -y pkg-config libhdf5-dev gcc

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY output/models /output/models
ENV MODEL_DIR /output/models
COPY app /app

# Expose the port the app runs on
EXPOSE 8000

# Specify the command to run on container start
CMD ["uvicorn", "app.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "app/gradio_main.py"]
#CMD ["python", "app/training.py"]
