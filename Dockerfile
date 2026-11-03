FROM python:3.11-slim

# Install system dependencies for OpenCV and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p captures/aligned_faces captures/face_crops captures/person_crops \
    captures/uploads data/face_embeddings data/registered_faces logs

# Expose nothing by default (headless mode)
CMD ["python", "main.py", "--no-show"]

