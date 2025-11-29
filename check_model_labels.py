import sys
sys.path.insert(0, 'src')

from transformers import AutoModelForTokenClassification

MODEL_PATH = "models/layoutlmv3"

print("Loading model...")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

print("\nModel classes/labels:")
print("="*60)

if hasattr(model.config, 'id2label'):
    for label_id, label_name in sorted(model.config.id2label.items()):
        print(f"  {label_id}: {label_name}")
    
    print(f"\nTotal: {len(model.config.id2label)} classes")
else:
    print("No id2label found in model config")

if hasattr(model.config, 'label2id'):
    print("\nLabel to ID mapping:")
    for label_name, label_id in sorted(model.config.label2id.items()):
        print(f"  {label_name}: {label_id}")
