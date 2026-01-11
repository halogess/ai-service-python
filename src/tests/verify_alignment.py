
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
        {'unit_id': '2_text', 'elem_id': 2, 'elem_seq': 2, 'elem_type': 'paragraph', 'text': 'This is specific text.', 'text_normalized': 'thisisspecifictext', 'is_cell': False},
        {'unit_id': '3_r0_c0', 'elem_id': 3, 'elem_seq': 3, 'elem_type': 'table', 'text': 'Cell 1', 'text_normalized': 'cell1', 'is_cell': True, 'row': 0, 'col': 0},
    ]
    
    # PDF Units (Source)
    # Split text for "Introduction" to test char alignment
    pdf_units = [
        {'unit_id': 'pdf_0', 'item_idx': 0, 'item_type': 'group', 'text': 'Intro', 'text_normalized': 'intro', 'bbox': [10, 10, 50, 20], 'is_cell': False},
        {'unit_id': 'pdf_1', 'item_idx': 1, 'item_type': 'group', 'text': 'duction', 'text_normalized': 'duction', 'bbox': [50, 10, 90, 20], 'is_cell': False},
        {'unit_id': 'pdf_2', 'item_idx': 2, 'item_type': 'group', 'text': 'This is specific text.', 'text_normalized': 'thisisspecifictext', 'bbox': [10, 30, 200, 40], 'is_cell': False},
        {'unit_id': 'pdf_3', 'item_idx': 3, 'item_type': 'table', 'text': 'Cell 1', 'text_normalized': 'cell1', 'bbox': [10, 50, 60, 60], 'is_cell': True},
    ]
    
    # Test perform_two_pass_alignment
    print("Running perform_two_pass_alignment...")
    result = service._perform_two_pass_alignment(pdf_units, openxml_units, min_openxml_idx=0)
    
    # Assertions
    alignments = result['final_alignments']
    print(f"Alignments found: {len(alignments)}")
    
    # 1. Check Introduction alignment
    intro_align = next((a for a in alignments if a['element_text'] == 'Introduction'), None)
    if intro_align:
        matched_texts = [u['text'] for u in intro_align['matched_pdf_units']]
        print(f"Introduction matched: {matched_texts}")
        if matched_texts == ['Intro', 'duction']:
            print("PASS: Introduction split-alignment verified.")
        else:
            print("FAIL: Introduction alignment match mismatch.")
    else:
        print("FAIL: Introduction element not aligned.")

    # 2. Check exact match
    exact_align = next((a for a in alignments if a['element_text'] == 'This is specific text.'), None)
    if exact_align:
        print("PASS: Exact match verified.")
    else:
        print("FAIL: Exact match not found.")

    # 3. Check table cell match
    cell_align = next((a for a in alignments if a['is_table'] and a['element_text'] == 'Cell 1'), None)
    if cell_align:
         print("PASS: Table cell alignment verified.")
    else:
         print("FAIL: Table cell alignment failed.")

    # Check unaligned
    unaligned = result['unaligned_final']
    if len(unaligned) == 0:
        print("PASS: All PDF units aligned.")
    else:
        print(f"FAIL: {len(unaligned)} unaligned PDF units found.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    run_test()
