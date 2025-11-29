import sys
sys.path.insert(0, 'src')

from pdf_processor import extract_layout_data_from_pdf
from layoutlm_processor import process_image_with_layoutlm
from PIL import Image
import json

if len(sys.argv) < 3:
    print("Usage: python debug_predictions.py <pdf_path> <image_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
image_path = sys.argv[2]

print("Extracting PDF data...")
pdf_data = extract_layout_data_from_pdf(pdf_path)
page_data = pdf_data[0]

print(f"\nPage has {len(page_data['words'])} words, {len(page_data['images'])} images")

print("\nProcessing with LayoutLM...")
result = process_image_with_layoutlm(image_path, page_data)

print(f"\nPredictions: {len(result['predictions'])}")
print(f"Image token map: {result['image_token_map']}")

# Group by label
from collections import Counter
label_counts = Counter([p['label'] for p in result['predictions']])
print(f"\nLabel distribution:")
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# Show predictions around images
print("\nPredictions around [IMG] tokens:")
for word_idx, img_idx in result['image_token_map'].items():
    print(f"\n  Image {img_idx} at word_idx {word_idx}:")
    for pred in result['predictions']:
        if abs(pred['token_id'] - word_idx) <= 5:
            print(f"    word_idx={pred['token_id']}: {pred['label']} (conf={pred['confidence']:.3f})")

# Save full result
with open('debug_predictions.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nFull result saved to debug_predictions.json")
