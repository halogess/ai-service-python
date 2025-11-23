FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (cached jika requirements tidak berubah)
RUN pip install --no-cache-dir -r requirements.txt

    # Copy source code
COPY src/ ./src/
COPY .env .

# Run the worker
CMD ["python", "src/main.py"]