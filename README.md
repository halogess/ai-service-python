# AI Service Python - Visual Worker

Background service untuk convert PDF ke images dan document analysis dengan Docling.

## Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-service-python
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env dengan kredensial database Anda
```

### 3. Run dengan Docker (Recommended)
```bash
docker-compose up -d
```

### 4. Manual Setup (Tanpa Docker)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run worker
python src/main.py
```

## Features

- Convert PDF to images (300 DPI)
- Document analysis dengan Docling
- Export ke JSON dan Markdown

## Monitoring

```bash
# Lihat logs
docker logs visual-worker -f

# Cek status
docker ps

# Restart
docker-compose restart
```