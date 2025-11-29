import sys
import os
sys.path.insert(0, 'src')

from pdf_processor import extract_layout_data_from_pdf, convert_pdf_to_images
from layoutlm_processor import process_document
import json

def test_local(pdf_path, output_dir="test_output"):
    """
    Test LayoutLM processing locally
    
    Usage:
        python test_local.py <path_to_pdf>
    """
    print(f"Testing with PDF: {pdf_path}\n")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Extract layout data
    print("Step 1: Extracting layout data from PDF...")
    pdf_data = extract_layout_data_from_pdf(pdf_path)
    
    for i, page in enumerate(pdf_data):
        print(f"  Page {i+1}: {len(page['words'])} words, {len(page['images'])} images")
    
    # Save extracted data
    with open(os.path.join(output_dir, "extracted_data.json"), 'w', encoding='utf-8') as f:
        json.dump(pdf_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_dir}/extracted_data.json\n")
    
    # Step 2: Convert PDF to images
    print("Step 2: Converting PDF to images...")
    image_paths = convert_pdf_to_images(pdf_path, images_dir)
    print(f"  Created {len(image_paths)} images in: {images_dir}\n")
    
    # Step 3: Process with LayoutLM
    print("Step 3: Processing with LayoutLMv3...")
    results = process_document(image_paths, output_dir=results_dir, pdf_text_data=pdf_data)
    
    # Step 4: Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for page_result in results:
        page_num = page_result['page_number']
        predictions = page_result['predictions']
        image_token_map = page_result.get('image_token_map', {})
        
        print(f"\nPage {page_num}:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Images detected: {len(image_token_map)}")
        
        # Count labels
        from collections import Counter
        label_counts = Counter([p['label'] for p in predictions])
        print(f"  Label distribution:")
        for label, count in label_counts.most_common():
            print(f"    {label}: {count}")
        
        # Show predictions for [IMG] tokens
        if image_token_map:
            print(f"  [IMG] token predictions:")
            for word_idx, img_idx in image_token_map.items():
                for pred in predictions:
                    if pred['token_id'] == word_idx:
                        print(f"    Image {img_idx}: {pred['label']} (confidence: {pred['confidence']:.3f})")
    
    # Save full results
    results_file = os.path.join(output_dir, "layoutlm_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}/")
    print(f"  - extracted_data.json: Raw PDF extraction")
    print(f"  - layoutlm_results.json: LayoutLM predictions")
    print(f"  - images/: Converted PDF pages")
    print(f"  - results/: Visualizations with bounding boxes")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_local.py <pdf_path> [output_dir]")
        print("\nExample:")
        print("  python test_local.py document.pdf")
        print("  python test_local.py document.pdf my_output")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_output"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    test_local(pdf_path, output_dir)
