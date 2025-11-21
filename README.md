# AI Service Python Worker

Background worker service untuk visual validation dokumen PDF menggunakan OCR dan Layout Model.

## Fitur

- Background worker tanpa HTTP API
- Membaca job dari database MySQL
- Pipeline visual validation:
  - PDF → gambar per halaman
  - OCR → teks + koordinat
  - Layout model → klasifikasi blok layout
  - Evaluasi aturan visual
- Menyimpan hasil ke database

## Struktur Proyek

```
src/
├── main.py                    # Entry point worker
├── config.py                  # Konfigurasi environment
├── db.py                      # SQLAlchemy setup
├── models/                    # ORM models
├── jobs/                      # Job processing
├── processing/                # Pipeline processing
└── utils/                     # Utilities
```

## Environment Variables

- `DB_HOST` - Host MySQL (default: mysql)
- `DB_PORT` - Port MySQL (default: 3306)
- `DB_NAME` - Nama database (default: db_korektor_buku)
- `DB_USER` - User database
- `DB_PASSWORD` - Password database
- `BASE_PATH` - Path shared storage (default: /data/cek-ta)
- `LOG_LEVEL` - Level logging (default: INFO)

## Menjalankan

### Development dengan Docker Compose

```bash
docker compose -f docker-compose.dev.yml up python-worker
```

### Manual

```bash
pip install -r requirements.txt
cd src
python main.py
```

## Dependencies

- SQLAlchemy 2.x - ORM
- PyMuPDF - PDF processing
- pytesseract - OCR
- transformers - Layout model
- mysql-connector-python - MySQL driver