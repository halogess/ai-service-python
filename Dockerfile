FROM python:3.11-slim

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

ARG UV_HTTP_TIMEOUT=10000
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

# Upgrade pip first, then install uv for dependency resolution
RUN pip install --upgrade pip \
    && pip install --no-cache-dir uv \
    && uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY .env .

# Run the worker
CMD ["python", "src/main.py"]
