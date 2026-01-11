
import sys
import os
from services.merging_extraction_service import MergingExtractionService
from services.alignment_service import AlignmentService
from app import create_app

def main():
    doc_id = 335
    page_num = 1
    
    from models import Dokumen
    
    app = create_app()
    with app.app_context():
        # Find a valid document
        doc = Dokumen.query.get(doc_id)
        if not doc:
            print(f"Document {doc_id} not found, fetching first available...")
            doc = Dokumen.query.first()
            if not doc:
                print("No documents found in database.")
                return
            doc_id = doc.dokumen_id
            print(f"Using Document ID: {doc_id}")
            
        print(f"--- Starting Extraction for Document {doc_id} Page {page_num} ---")
        
        # 1. Extraction
        extractor = MergingExtractionService()
        merged_data = extractor.extract_and_process(doc_id, page_num)
        
        stats = merged_data.get('stats', {})
        print(f"Extraction stats: {stats}")
        
        items = merged_data.get('items', [])
        if not items:
            print("No items extracted!")
            return

        print(f"Extracted {len(items)} items.")
        
        # 2. Alignment
        aligner = AlignmentService()
        alignment_result = aligner.align(
            doc_id, 
            page_num, 
            items,
            merged_data['width'],
            merged_data['height'],
            1 # Assuming 1 page for test or fetch from doc
        )
        
        alignments = alignment_result.get('alignments', [])
        print(f"Alignment complete. Found {len(alignments)} matches.")
        
        print("\n--- Sample Alignments ---")
        for i, align in enumerate(alignments[:5]):
            print(f"Match {i+1}: PDF '{align['text'][:30]}...' <--> XML ID {align['element_id']} (Score: {align['score']:.2f})")

if __name__ == "__main__":
    main()
