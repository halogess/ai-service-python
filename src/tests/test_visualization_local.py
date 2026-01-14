import sys
import os
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.visualization_service import VisualizationService

def test_visualization_local():
    print("Testing Visualization Locally...")
    
    # Path to local PDF
    pdf_path = os.path.abspath("bab2.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        # Try to find it if current dir is different
        if os.path.exists("../bab2.pdf"):
            pdf_path = os.path.abspath("../bab2.pdf")
        elif os.path.exists("e:/ai-service-python/bab2.pdf"):
             pdf_path = "e:/ai-service-python/bab2.pdf"
        else:
            print("Cannot find bab2.pdf, aborting.")
            return

    print(f"Using PDF: {pdf_path}")

    # Mock data
    doc_id = 999
    alignments = [
        {'merged_bbox': [50, 50, 200, 100], 'element_sequence': 1, 'is_text_part': True, 'element_type': 'p'},
        {'merged_bbox': [50, 120, 200, 200], 'element_sequence': 2, 'is_image_part': True, 'element_type': 'figure'}
    ]
    
    fused_results = [
        {'bbox': [50, 50, 200, 100], 'label': 'paragraph', 'element_sequence': 1, 'overlap': 1.0},
        {'bbox': [50, 120, 200, 200], 'label': 'picture', 'element_sequence': 2, 'overlap': 1.0}
    ]
    
    vis_service = VisualizationService(output_dir='visualization_output_local')
    
    # Clean previous output
    if os.path.exists('visualization_output_local'):
        shutil.rmtree('visualization_output_local')
        
    saved_paths = vis_service.visualize_page(
        pdf_path=pdf_path,
        page_num=0,
        alignments=alignments,
        fused_results=fused_results,
        doc_id=doc_id
    )
    
    print("\nSaved paths:")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")
        
    # Assertions
    expected_keys = ['fused']
    keys = list(saved_paths.keys())
    
    if keys == expected_keys:
        print("\nSUCCESS: Only 'fused' image was saved.")
    else:
        print(f"\nFAILURE: Expected {expected_keys}, got {keys}")
        sys.exit(1)
        
    # Verify file name
    fused_path = saved_paths['fused']
    if not fused_path.endswith("_fused.png"):
         print(f"FAILURE: File name does not end with _fused.png: {fused_path}")
         sys.exit(1)
    
    if "fusion.png" in fused_path:
        print(f"FAILURE: File name contains fusion.png (old name): {fused_path}")
        sys.exit(1)
        
    print("File name verification passed.")

if __name__ == "__main__":
    test_visualization_local()
