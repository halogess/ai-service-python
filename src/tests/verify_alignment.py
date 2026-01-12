
import sys
import os
import json
from unittest.mock import MagicMock

# Ensure src is in pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock database and models to prevent connection attempts
sys.modules['database'] = MagicMock()
sys.modules['models'] = MagicMock()
sys.modules['models.DokumenElemen'] = MagicMock()
sys.modules['models.DokumenSection'] = MagicMock()
sys.modules['models.DokumenPart'] = MagicMock()

from services.alignment_service import AlignmentService

def run_test():
    print("Verifying AlignmentService logic...")
    service = AlignmentService()
    
    # Mock Data
    # OpenXML Units (Target)
    openxml_units = [
        {'unit_id': '1_text', 'elem_id': 1, 'elem_seq': 1, 'elem_type': 'paragraph', 'text': 'Introduction', 'text_normalized': 'introduction', 'is_cell': False},
        # Shape that should now be [IMG]
        {'unit_id': '2_shape', 'elem_id': 2, 'elem_seq': 2, 'elem_type': 'shape', 'text': '[IMG]', 'text_normalized': '[img]', 'is_cell': False, 'has_shape': True},
        {'unit_id': '3_text', 'elem_id': 3, 'elem_seq': 3, 'elem_type': 'paragraph', 'text': 'End', 'text_normalized': 'end', 'is_cell': False},
    ]
    
    # PDF Units (Source)
    pdf_units = [
        {'unit_id': 'pdf_0', 'item_idx': 0, 'item_type': 'group', 'text': 'Intro', 'text_normalized': 'intro', 'bbox': [10, 10, 50, 20], 'is_cell': False},
        {'unit_id': 'pdf_1', 'item_idx': 1, 'item_type': 'group', 'text': 'duction', 'text_normalized': 'duction', 'bbox': [50, 10, 90, 20], 'is_cell': False},
        # Shape with image but empty text (The "Ghost Image")
        # alignment_service should now convert this to [IMG] internally if logic works
        # BUT verify_alignment calls _perform_two_pass_alignment directly with ALREADY FLATTENED units if we pass them as pdf_units.
        # Wait, _flatten_extraction_items is called by `align`. 
        # Here we are mocking `pdf_units` which are the OUTPUT of `_flatten_extraction_items`.
        # So we should verify `_flatten_extraction_items` logic separately OR mock the flattened input properly.
        
        # Let's test _flatten_extraction_items logic first?
    ]
    
    # Test _flatten_extraction_items logic for Ghost Image
    print("\nTesting _flatten_extraction_items for Ghost Image...")
    raw_shape = {
        'type': 'shape', 'bbox': [100, 100, 200, 200],
        'data': {'text': '   ', 'image_bbox': [100, 100, 200, 200]} # Whitespace text but has image
    }
    flattened = service._flatten_extraction_items([raw_shape])
    print(f"Flattened result: {flattened}")
    
    if len(flattened) == 1 and flattened[0]['text'] == '[IMG]':
        print("PASS: Ghost Image shape preserved and converted to [IMG].")
        pdf_units.append(flattened[0]) # Add to our manual list for alignment test
    else:
        print("FAIL: Ghost Image shape was discarded or not converted.")

    pdf_units.append({'unit_id': 'pdf_3', 'item_idx': 3, 'item_type': 'group', 'text': 'End', 'text_normalized': 'end', 'bbox': [10, 300, 50, 320], 'is_cell': False})

    # Test perform_two_pass_alignment
    print("\nRunning perform_two_pass_alignment...")
    # Fix unit_ids for pdf_units since we appended manually
    for i, u in enumerate(pdf_units): u['unit_id'] = f'pdf_{i}'
    
    result = service._perform_two_pass_alignment(pdf_units, openxml_units, min_openxml_idx=0)
    
    # Assertions
    alignments = result['final_alignments']
    print(f"Alignments found: {len(alignments)}")
    
    # Check Shape alignment
    shape_align = next((a for a in alignments if a['element_type'] == 'shape'), None)
    if shape_align:
        matched = shape_align['matched_pdf_units']
        print(f"Shape matched: {len(matched)} units. Text: {[u['text'] for u in matched]}")
        if len(matched) > 0 and matched[0]['text'] == '[IMG]':
            print("PASS: Shape aligned correctly to [IMG] unit.")
        else:
            print("FAIL: Shape matched incorrect unit.")
    else:
        print("FAIL: Shape element not aligned.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    run_test()
