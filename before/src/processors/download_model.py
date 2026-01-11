from transformers import AutoModelForTokenClassification, AutoProcessor

MODEL_NAME = "Kwan0/layoutlmv3-base-finetune-DocLayNet-100k"
MODEL_PATH = "models/layoutlmv3"

print(f"Downloading model: {MODEL_NAME}")
print("This may take a few minutes...")

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
print("Done!")
