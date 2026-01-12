
import sys
import os
import io
from unittest.mock import MagicMock, patch

# Adjust path to src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock database
mock_db = MagicMock()
mock_doc = MagicMock()
mock_doc.dokumen_pdf_path = "validasi-ta/dokumen/222117032/332/pdf/BAB II.pdf"
mock_db.query.return_value.get.return_value = mock_doc

# Mock SessionLocal
with patch('services.merging_extraction_service.SessionLocal', return_value=mock_db):
    try:
        from services.merging_extraction_service import MergingExtractionService
        
        print("Initializing Service...")
        service = MergingExtractionService()
        
        # Mock dependencies to focus on MergingExtractionService logic
        # Mock Docling
        service.docling_service.classify_document = MagicMock(return_value={
            'success': True,
            'predictions_by_page': {
                '1': [{'label': 'text', 'bbox': [100, 100, 200, 200], 'text': 'Mock Text'}]
            }
        })
        
        # Mock Alignment (return success)
        service.alignment_service.align = MagicMock(return_value={
            'success': True,
            'final_alignments': [{
                'element_id': 1,
                'merged_bbox': [100, 100, 200, 200],
                'matched_pdf_units': [],
                'element_sequence': 1,
                'element_type': 'paragraph'
            }],
            'max_openxml_idx': 10
        })
        
        # Mock PDFExtractor (partially)
        # We need it to return valid execution flow.
        # If we can't open real PDF, we simulate it?
        # But MergingExtractionService orchestrates it. 
        # If we let it run with real PDF path (if accessible), it's better test.
        # But environment var VOLUME_BASE_PATH might differ.
        
        # Set env var for test
        os.environ["VOLUME_BASE_PATH"] = "E:/docker-volumes"
        
        print("Running process_document...")
        # doc_id=332, generate_visualizations=True, save_to_db=False, output_dir=...
        output_dir = "tests/output_debug"
        os.makedirs(output_dir, exist_ok=True)
        
        success = service.process_document(
            doc_id=332,
            generate_visualizations=True,
            save_to_db=False,
            output_dir=output_dir
        )
        
        if success:
            print("SUCCESS: process_document returned True")
        else:
            print("FAILURE: process_document returned False")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL EXCEPTION: {e}")
