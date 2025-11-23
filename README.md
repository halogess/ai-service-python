# AI Service Python - Visual Worker

Background service untuk memproses dokumen dengan LayoutLMv3.

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

### 3. Download Model
```bash
# Aktifkan virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Download model (sekali saja)
python download_model.py
```

### 4. Run dengan Docker (Recommended)
```bash
docker-compose up -d
```

### 4. Manual Setup (Tanpa Docker)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download model
python -c "from transformers import AutoModelForTokenClassification, AutoProcessor; \
    model = AutoModelForTokenClassification.from_pretrained('Kwan0/layoutlmv3-base-finetune-DocLayNet-100k'); \
    processor = AutoProcessor.from_pretrained('Kwan0/layoutlmv3-base-finetune-DocLayNet-100k'); \
    model.save_pretrained('models/layoutlmv3'); \
    processor.save_pretrained('models/layoutlmv3')"

# Run worker
python src/main.py
```

## Model

Model: `Kwan0/layoutlmv3-base-finetune-DocLayNet-100k`
- Download sekali dengan `python download_model.py`
- Disimpan di `models/layoutlmv3/` (lokal)
- Docker akses via volume mount
- Bisa di-commit ke git atau di-gitignore (opsional)

## Monitoring

```bash
# Lihat logs
docker logs visual-worker -f

# Cek status
docker ps

# Restart
docker-compose restart
```