FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY generate_debug_html.py .
COPY .env .

# Run the worker
CMD ["python3.11", "src/main.py"]