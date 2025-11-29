import sys
sys.path.insert(0, 'src')

from pdf_processor import extract_layout_data_from_pdf
import json

if len(sys.argv) < 2:
    print("Usage: python test_image_detection.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Testing image detection on: {pdf_path}\n")

data = extract_layout_data_from_pdf(pdf_path)

for page_num, page_data in enumerate(data):
    print(f"=== PAGE {page_num + 1} ===")
    print(f"Words: {len(page_data['words'])}")
    print(f"Blocks: {len(page_data['blocks'])}")
    print(f"Images: {len(page_data['images'])}")
    
    if page_data['images']:
        print("\nImage details:")
        for idx, img in enumerate(page_data['images']):
            print(f"  Image {idx + 1}: bbox={img['bbox']}, block_no={img['block_no']}")
    
    print()

# Save to JSON for inspection
output_file = "test_extraction_output.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Full data saved to: {output_file}")
