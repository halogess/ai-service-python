
import sys
import os
import unittest.mock
from unittest.mock import MagicMock

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from services.merging_extraction_service import MergingExtractionService
from services.alignment_service import AlignmentService
from models import Dokumen, DokumenElemen
import database

def main():
    # Hardcoded test file found in workspace
    TEST_PDF_PATH = os.path.join(os.getcwd(), 'bab2.pdf')
    DOC_ID = 335
    PAGE_NUM = 1
    
    print(f"--- Initializing Test with {TEST_PDF_PATH} ---")
    
    # Mocking Database Session for Extraction
    mock_doc = MagicMock(spec=Dokumen)
    mock_doc.dokumen_id = DOC_ID
    mock_doc.dokumen_pdf_path = TEST_PDF_PATH
    
    mock_session = MagicMock()
    mock_session.query.return_value.get.return_value = mock_doc
    
    with unittest.mock.patch('services.merging_extraction_service.SessionLocal', return_value=mock_session):
        print("--- Starting Extraction (Mocked DB) ---")
        
        try:
            extractor = MergingExtractionService()
            merged_data = extractor.extract_and_process(DOC_ID, PAGE_NUM)
            
            stats = merged_data.get('stats', {})
            print(f"Extraction stats: {stats}")
            
            items = merged_data.get('items', [])
            if not items:
                print("No items extracted!")
            else:
                print(f"Extracted {len(items)} items.")
                # Print first few items to verify
                for i, item in enumerate(items[:3]):
                    print(f"Item {i}: Type={item.get('type')}, BBox={item.get('bbox')}")
                    if item.get('type') == 'group':
                         print(f"   Text: {item['data'].get('text')[:50]}...")

        except Exception as e:
            print(f"Extraction Failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n--- Starting Alignment (Mocked DB - No OpenXML) ---")
    
    mock_session_align = MagicMock()
    # Mock OpenXML elements as empty list
    mock_session_align.query.return_value.join.return_value.filter.return_value.order_by.return_value.all.return_value = []
    
    # Mock Sections with real integers for margin logic
    mock_section = MagicMock()
    mock_section.dsec_margin_top_twips = 1440  # 1 inch
    mock_section.dsec_margin_bottom_twips = 1440
    mock_section.dsec_page_height_twips = 15840 # 11 inches
    
    # Patch _get_doc_sections directly to avoid SQLAlchemy mock chain hell
    with unittest.mock.patch('services.alignment_service.SessionLocal', return_value=mock_session_align), \
         unittest.mock.patch.object(AlignmentService, '_get_doc_sections', return_value=[mock_section]):
         try:
            aligner = AlignmentService()
            # We use the items extracted above
            alignment_result = aligner.align(
                DOC_ID, 
                PAGE_NUM, 
                items if 'items' in locals() else [],
                merged_data['width'] if 'merged_data' in locals() else 595,
                merged_data['height'] if 'merged_data' in locals() else 842,
                1
            )
            
            alignments = alignment_result.get('alignments', [])
            print(f"Alignment complete (with 0 elements to match). Result count: {len(alignments)}")
            
         except Exception as e:
            print(f"Alignment Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
