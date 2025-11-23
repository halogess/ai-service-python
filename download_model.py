"""
Download LayoutLMv3 model locally.

Gunakan script ini untuk:
- Setup pertama kali
- Re-download model jika corrupt
- Setup di server/laptop baru

Cara pakai:
    python download_model.py

Model akan tersimpan di folder models/layoutlmv3/
"""

from transformers import AutoModelForTokenClassification, AutoProcessor
import os

MODEL_PATH = "models/layoutlmv3"
MODEL_NAME = "Kwan0/layoutlmv3-base-finetune-DocLayNet-100k"

print(f"Downloading model from {MODEL_NAME}...")

# Create directory
os.makedirs(MODEL_PATH, exist_ok=True)

# Download model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Save locally
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)

print(f"Model downloaded and saved to {MODEL_PATH}")
print("Ready to use with Docker!")