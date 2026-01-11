FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies one by one
RUN pip install --no-cache-dir --timeout=1000 sqlalchemy==2.0.44
RUN pip install --no-cache-dir --timeout=1000 pymysql==1.1.2
RUN pip install --no-cache-dir --timeout=1000 python-dotenv
RUN pip install --no-cache-dir --timeout=1000 pillow
RUN pip install --no-cache-dir --timeout=1000 pymupdf
RUN pip install --no-cache-dir --timeout=1000 rapidfuzz
RUN pip install --no-cache-dir --timeout=10000 docling

# Copy source code
COPY src/ ./src/
COPY .env .

# Run the worker
CMD ["python", "src/main.py"]