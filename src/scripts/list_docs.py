
import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import SessionLocal
from models import Dokumen

def list_docs():
    db = SessionLocal()
    try:
        docs = db.query(Dokumen).limit(5).all()
        for d in docs:
            print(f"ID: {d.dokumen_id}, PDF: {d.dokumen_pdf_path}")
    finally:
        db.close()

if __name__ == "__main__":
    list_docs()
